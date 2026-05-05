"""MiroFish Offline API Wrapper – Flask Proxy Server.

A REST server on port 5050 that orchestrates the full MiroFish simulation
pipeline (6 stages) in background threads and exposes a simple API for the
frontend.
"""

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# ======================================================================
# MiroFish low-level client
# ======================================================================

MIROFISH_BASE_URL = os.getenv("MIROFISH_BASE_URL", "http://localhost:5001")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "3600"))
SIM_RUN_TIMEOUT = int(os.getenv("SIM_RUN_TIMEOUT", "21600"))  # 6h for multi-agent simulation runs


class MiroFishAPIError(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class MiroFishClient:
    def __init__(self, base_url: str = MIROFISH_BASE_URL, timeout: int = LLM_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.request(method, self._url(path), **kwargs)
        if not resp.ok:
            raise MiroFishAPIError(resp.status_code, resp.text)
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct:
            return resp.json()
        return resp.text

    # Health
    def health(self) -> Any:
        return self._request("GET", "/health")

    # Graph / Ontology
    def generate_ontology(self, simulation_requirement: str, project_name: str) -> Any:
        return self._request(
            "POST",
            "/api/graph/ontology/generate",
            data={
                "simulation_requirement": simulation_requirement,
                "project_name": project_name,
            },
            files={
                "files": (
                    "seed_document.txt",
                    simulation_requirement.encode("utf-8"),
                    "text/plain",
                ),
            },
        )

    def build_graph(self, project_id: str, **kwargs) -> Any:
        payload = {"project_id": project_id, **kwargs}
        return self._request("POST", "/api/graph/build", json=payload)

    def get_task_status(self, task_id: str) -> Any:
        return self._request("GET", f"/api/graph/task/{task_id}")

    def wait_for_task(self, task_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None) -> Any:
        elapsed = 0.0
        while True:
            result = self.get_task_status(task_id)
            data = result.get("data", {}) if isinstance(result, dict) else {}
            status = data.get("status", "")
            if status == "completed":
                return result
            if status == "failed":
                error_msg = data.get("error", "Task failed")
                raise MiroFishAPIError(500, error_msg)
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(f"Task {task_id} did not finish within {timeout}s")
            time.sleep(poll_interval)
            elapsed += poll_interval

    # Simulation
    def create_simulation(self, project_id: str, **kwargs) -> Any:
        payload = {"project_id": project_id, **kwargs}
        return self._request("POST", "/api/simulation/create", json=payload)

    def prepare_simulation(self, simulation_id: str, **kwargs) -> Any:
        payload = {"simulation_id": simulation_id, **kwargs}
        return self._request("POST", "/api/simulation/prepare", json=payload)

    def get_prepare_status(self, simulation_id: str, task_id: str | None = None) -> Any:
        payload: dict[str, Any] = {"simulation_id": simulation_id}
        if task_id:
            payload["task_id"] = task_id
        return self._request("POST", "/api/simulation/prepare/status", json=payload)

    def wait_for_prepare(
        self, simulation_id: str, task_id: str | None = None,
        poll_interval: float = 5.0, timeout: Optional[float] = None,
    ) -> Any:
        elapsed = 0.0
        while True:
            result = self.get_prepare_status(simulation_id, task_id)
            data = result.get("data", {}) if isinstance(result, dict) else {}
            status = data.get("status", "")
            if status in ("ready", "completed"):
                return result
            if status == "failed":
                raise MiroFishAPIError(500, data.get("message", "Prepare failed"))
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(f"Prepare for {simulation_id} did not finish within {timeout}s")
            time.sleep(poll_interval)
            elapsed += poll_interval

    def generate_profiles(self, graph_id: str, **kwargs) -> Any:
        payload = {"graph_id": graph_id, **kwargs}
        return self._request("POST", "/api/simulation/generate-profiles", json=payload)

    def start_simulation(self, simulation_id: str, **kwargs) -> Any:
        payload = {"simulation_id": simulation_id, **kwargs}
        return self._request("POST", "/api/simulation/start", json=payload)

    def get_run_status(self, simulation_id: str) -> Any:
        return self._request("GET", f"/api/simulation/{simulation_id}/run-status")

    def wait_for_simulation(
        self, simulation_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None
    ) -> Any:
        elapsed = 0.0
        while True:
            result = self.get_run_status(simulation_id)
            data = result.get("data", {}) if isinstance(result, dict) else {}
            runner_status = data.get("runner_status", "")
            if runner_status in ("completed", "stopped", "failed", "error"):
                return result
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    f"Simulation {simulation_id} did not finish within {timeout}s"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval

    # Report
    def generate_report(self, simulation_id: str, **kwargs) -> Any:
        payload = {"simulation_id": simulation_id, **kwargs}
        return self._request("POST", "/api/report/generate", json=payload)

    def report_chat(self, simulation_id: str, message: str, chat_history: list | None = None) -> Any:
        payload = {"simulation_id": simulation_id, "message": message, "chat_history": chat_history or []}
        return self._request("POST", "/api/report/chat", json=payload)


# ======================================================================
# 27 EU country profiles (Eurostat-based reference data)
# ======================================================================

EU_COUNTRIES: list[dict] = [
    {"code": "AT", "name": "Austria",        "population": 9104772,  "gdp_per_capita": 53637, "unemployment_rate": 5.0, "capital": "Vienna"},
    {"code": "BE", "name": "Belgium",        "population": 11686140, "gdp_per_capita": 49540, "unemployment_rate": 5.5, "capital": "Brussels"},
    {"code": "BG", "name": "Bulgaria",       "population": 6447710,  "gdp_per_capita": 13980, "unemployment_rate": 4.2, "capital": "Sofia"},
    {"code": "HR", "name": "Croatia",        "population": 3862305,  "gdp_per_capita": 18570, "unemployment_rate": 6.1, "capital": "Zagreb"},
    {"code": "CY", "name": "Cyprus",         "population": 920701,   "gdp_per_capita": 31450, "unemployment_rate": 6.0, "capital": "Nicosia"},
    {"code": "CZ", "name": "Czech Republic", "population": 10827529, "gdp_per_capita": 27870, "unemployment_rate": 2.6, "capital": "Prague"},
    {"code": "DK", "name": "Denmark",        "population": 5932654,  "gdp_per_capita": 67790, "unemployment_rate": 4.8, "capital": "Copenhagen"},
    {"code": "EE", "name": "Estonia",        "population": 1365884,  "gdp_per_capita": 28350, "unemployment_rate": 6.4, "capital": "Tallinn"},
    {"code": "FI", "name": "Finland",        "population": 5563970,  "gdp_per_capita": 53650, "unemployment_rate": 7.2, "capital": "Helsinki"},
    {"code": "FR", "name": "France",         "population": 68042591, "gdp_per_capita": 44408, "unemployment_rate": 7.3, "capital": "Paris"},
    {"code": "DE", "name": "Germany",        "population": 84482267, "gdp_per_capita": 51383, "unemployment_rate": 3.0, "capital": "Berlin"},
    {"code": "GR", "name": "Greece",         "population": 10394055, "gdp_per_capita": 20867, "unemployment_rate": 11.0, "capital": "Athens"},
    {"code": "HU", "name": "Hungary",        "population": 9597085,  "gdp_per_capita": 18728, "unemployment_rate": 4.1, "capital": "Budapest"},
    {"code": "IE", "name": "Ireland",        "population": 5194336,  "gdp_per_capita": 103685, "unemployment_rate": 4.3, "capital": "Dublin"},
    {"code": "IT", "name": "Italy",          "population": 58850717, "gdp_per_capita": 35657, "unemployment_rate": 7.6, "capital": "Rome"},
    {"code": "LV", "name": "Latvia",         "population": 1883008,  "gdp_per_capita": 21148, "unemployment_rate": 6.8, "capital": "Riga"},
    {"code": "LT", "name": "Lithuania",      "population": 2857279,  "gdp_per_capita": 24030, "unemployment_rate": 6.9, "capital": "Vilnius"},
    {"code": "LU", "name": "Luxembourg",     "population": 660809,   "gdp_per_capita": 126426, "unemployment_rate": 4.9, "capital": "Luxembourg"},
    {"code": "MT", "name": "Malta",          "population": 542051,   "gdp_per_capita": 33486, "unemployment_rate": 3.0, "capital": "Valletta"},
    {"code": "NL", "name": "Netherlands",    "population": 17811291, "gdp_per_capita": 57768, "unemployment_rate": 3.5, "capital": "Amsterdam"},
    {"code": "PL", "name": "Poland",         "population": 36753736, "gdp_per_capita": 18321, "unemployment_rate": 2.8, "capital": "Warsaw"},
    {"code": "PT", "name": "Portugal",       "population": 10467366, "gdp_per_capita": 25065, "unemployment_rate": 6.5, "capital": "Lisbon"},
    {"code": "RO", "name": "Romania",        "population": 19038098, "gdp_per_capita": 15792, "unemployment_rate": 5.4, "capital": "Bucharest"},
    {"code": "SK", "name": "Slovakia",       "population": 5428792,  "gdp_per_capita": 21088, "unemployment_rate": 5.8, "capital": "Bratislava"},
    {"code": "SI", "name": "Slovenia",       "population": 2116972,  "gdp_per_capita": 29291, "unemployment_rate": 3.7, "capital": "Ljubljana"},
    {"code": "ES", "name": "Spain",          "population": 48059777, "gdp_per_capita": 30996, "unemployment_rate": 11.7, "capital": "Madrid"},
    {"code": "SE", "name": "Sweden",         "population": 10551707, "gdp_per_capita": 55873, "unemployment_rate": 7.5, "capital": "Stockholm"},
]

_COUNTRY_BY_CODE: dict[str, dict] = {c["code"]: c for c in EU_COUNTRIES}

# ======================================================================
# Named-entity narratives per country (for NER / knowledge-graph seeding)
# ======================================================================
# Each entry is a short news-style paragraph that mentions 15-20 real
# organisations, companies, institutions, media outlets, and key roles.
# MiroFish NER uses these to populate the knowledge graph with agents.

_COUNTRY_ENTITIES: dict[str, str] = {
    "AT": (
        "In Vienna, the Federation of Austrian Industries (Industriellenvereinigung) "
        "convened a roundtable with OMV AG, voestalpine AG, and Erste Group Bank AG to "
        "discuss the economic outlook. The Austrian National Bank (OeNB) published new "
        "projections while the Austrian Institute of Economic Research (WIFO) and the "
        "Institute for Advanced Studies (IHS) provided competing analyses. Federal "
        "Chancellor Karl Nehammer's office coordinated with the Federal Ministry of "
        "Finance (BMF) and the Austrian Economic Chambers (WKO). Coverage in Der "
        "Standard and Die Presse highlighted statements by the CEO of Verbund AG and "
        "the rector of TU Wien. The Austrian Trade Union Federation (ÖGB) and the "
        "Chamber of Labour (Arbeiterkammer) weighed in on labour market implications, "
        "while Red Bull GmbH and Raiffeisen Bank International announced new investment "
        "plans. The Austrian Research Promotion Agency (FFG) confirmed additional "
        "funding for the Austrian Academy of Sciences (ÖAW)."
    ),
    "BE": (
        "The Federation of Enterprises in Belgium (FEB/VBO) met with representatives "
        "from AB InBev, UCB SA, and KBC Group in Brussels. The National Bank of Belgium "
        "(NBB) released its financial stability report as the Federal Planning Bureau "
        "updated growth forecasts. Belgium's Deputy Prime Minister coordinated with the "
        "Federal Public Service Economy (FPS Economy) and the Flemish Department of "
        "Economy (VLAIO). Le Soir and De Tijd reported on statements by the CEO of "
        "Proximus and the managing director of Solvay SA. The Belgian General Federation "
        "of Labour (ABVV/FGTB) raised concerns about automation while Umicore NV and "
        "Ageas SA announced quarterly results. Research institutions including imec and "
        "the Université catholique de Louvain (UCLouvain) presented findings on digital "
        "transformation. The Flemish employers' organisation Voka and the port authority "
        "Port of Antwerp-Bruges discussed logistics competitiveness."
    ),
    "BG": (
        "In Sofia, the Bulgarian Industrial Capital Association (BICA) hosted a forum "
        "with Lukoil Neftochim Burgas, Aurubis Bulgaria, and Sopharma AD. The Bulgarian "
        "National Bank (BNB) reported on euro-adoption readiness while the Center for "
        "Economic Strategies and Competitiveness (CESC) released an investment climate "
        "survey. The Ministry of Economy and Industry coordinated with the Bulgarian "
        "Chamber of Commerce and Industry (BCCI). Capital Weekly and Dnevnik covered "
        "statements from the CEO of Bulgarian Energy Holding (BEH) and the rector of "
        "Sofia University St. Kliment Ohridski. The Confederation of Independent Trade "
        "Unions in Bulgaria (CITUB) and the Bulgarian Academy of Sciences (BAS) "
        "discussed workforce development. Eurohold Bulgaria, DSK Bank, and the Bulgarian "
        "Development Bank announced infrastructure financing. The Agency for Small and "
        "Medium-sized Enterprises (BSMEPA) and the National Statistical Institute (NSI) "
        "released new data on regional disparities."
    ),
    "HR": (
        "The Croatian Employers' Association (HUP) organised a conference in Zagreb "
        "with representatives from INA Group, Atlantic Grupa, and Podravka dd. The "
        "Croatian National Bank (HNB) assessed eurozone integration effects while the "
        "Institute of Economics Zagreb published its economic outlook. The Ministry of "
        "Economy and Sustainable Development coordinated with the Croatian Chamber of "
        "Economy (HGK). Jutarnji List and Večernji List reported on statements by the "
        "CEO of Hrvatski Telekom and the president of the Croatian Academy of Sciences "
        "and Arts (HAZU). The Union of Autonomous Trade Unions of Croatia (SSSH) raised "
        "wage-growth concerns. Rimac Technology, Infobip, and Valamar Riviera presented "
        "expansion plans. The Ruđer Bošković Institute and the Faculty of Electrical "
        "Engineering and Computing at the University of Zagreb released a joint study "
        "on digital infrastructure. The Croatian Bureau of Statistics (DZS) updated "
        "labour force data."
    ),
    "CY": (
        "The Cyprus Chamber of Commerce and Industry (CCCI) held talks in Nicosia with "
        "Bank of Cyprus, Hellenic Bank, and Cyprus Telecommunications Authority (CYTA). "
        "The Central Bank of Cyprus published its lending survey while the Economics "
        "Research Centre of the University of Cyprus released an inflation forecast. The "
        "Ministry of Finance coordinated policy with the Cyprus Investment Promotion "
        "Agency (CIPA). The Cyprus Mail and Phileleftheros covered statements by the CEO "
        "of Eurobank Cyprus and the chair of the Cyprus Securities and Exchange "
        "Commission (CySEC). The Cyprus Workers' Confederation (SEK) and the Pancyprian "
        "Federation of Labour (PEO) debated minimum-wage adjustments. The Cyprus "
        "Institute, the CYENS Centre of Excellence, and the Cyprus Energy Regulatory "
        "Authority (CERA) discussed the energy transition. Cyprus Airways, Wargaming "
        "Nicosia, and the Limassol Port Authority commented on trade competitiveness."
    ),
    "CZ": (
        "The Confederation of Industry of the Czech Republic (SP ČR) convened in Prague "
        "with Škoda Auto, ČEZ Group, and Agrofert. The Czech National Bank (ČNB) "
        "tightened monetary guidance while the Czech Fiscal Council published debt "
        "sustainability projections. The Ministry of Industry and Trade coordinated with "
        "the Czech-Moravian Confederation of Trade Unions (ČMKOS). Hospodářské noviny "
        "and Lidové noviny reported on the CEO of Komerční banka and the rector of "
        "Charles University. PPF Group, Energo-Pro, and Avast Software outlined their "
        "growth strategies. The Czech Academy of Sciences (AV ČR) and the Czech "
        "Technical University (ČVUT) presented joint research on advanced manufacturing. "
        "The Technology Agency of the Czech Republic (TA ČR) and CzechInvest announced "
        "new incentives for semiconductor supply-chain investments."
    ),
    "DK": (
        "The Confederation of Danish Industry (DI) held its annual summit in Copenhagen "
        "with executives from Novo Nordisk, A.P. Møller-Mærsk, and Vestas Wind Systems. "
        "The Danmarks Nationalbank published a financial-stability assessment while the "
        "Danish Economic Councils (De Økonomiske Råd) released productivity projections. "
        "The Ministry of Industry, Business and Financial Affairs coordinated with the "
        "Danish Chamber of Commerce (Dansk Erhverv). Berlingske and Politiken covered "
        "statements by the CEO of Ørsted and the president of the Technical University "
        "of Denmark (DTU). The Danish Trade Union Confederation (FH) discussed green "
        "transition labour needs. Carlsberg Group, Pandora A/S, and Danske Bank outlined "
        "sustainability commitments. The Novo Nordisk Foundation and the Danish Agency "
        "for Higher Education and Science announced research funding. Copenhagen "
        "Infrastructure Partners (CIP) and the Danish Energy Agency discussed offshore "
        "wind expansion."
    ),
    "EE": (
        "The Estonian Employers' Confederation (Tööandjate Keskliit) met in Tallinn "
        "with Bolt Technology, Wise (TransferWise), and Eesti Energia. The Bank of "
        "Estonia (Eesti Pank) published its economic forecast while the Estonian "
        "Institute of Economic Research (EKI) assessed digital-economy growth. The "
        "Ministry of Economic Affairs and Communications coordinated with the Estonian "
        "Chamber of Commerce and Industry (EKTK). Postimees and ERR News reported on "
        "statements by the CEO of Telia Eesti and the rector of Tallinn University of "
        "Technology (TalTech). The Estonian Trade Union Confederation (EAKL) discussed "
        "remote-work policies. Skeleton Technologies, Nortal, and Swedbank Estonia "
        "announced R&D investments. The National Institute of Chemical Physics and "
        "Biophysics (KBFI) and the e-Governance Academy presented findings on digital "
        "public services. Enterprise Estonia (EAS) and the Estonian Research Council "
        "(ETAg) launched joint innovation calls."
    ),
    "FI": (
        "The Confederation of Finnish Industries (EK) hosted a policy forum in Helsinki "
        "with Nokia, Neste, and UPM-Kymmene. The Bank of Finland (Suomen Pankki) "
        "adjusted its inflation outlook while the Research Institute of the Finnish "
        "Economy (ETLA) published a competitiveness report. The Ministry of Economic "
        "Affairs and Employment coordinated with Business Finland and the Finnish "
        "Chamber of Commerce (Keskuskauppakamari). Helsingin Sanomat and Kauppalehti "
        "reported on the CEO of Fortum and the president of VTT Technical Research "
        "Centre of Finland. The Central Organisation of Finnish Trade Unions (SAK) "
        "raised concerns about AI-driven displacement. KONE Corporation, Wärtsilä, and "
        "Nordea Bank Finland discussed supply-chain resilience. Aalto University and the "
        "Finnish Institute for Health and Welfare (THL) presented demographic impact "
        "research. The Academy of Finland and Sitra (the Finnish Innovation Fund) "
        "announced clean-energy funding initiatives."
    ),
    "FR": (
        "In Paris, the Movement of the Enterprises of France (MEDEF) convened with "
        "TotalEnergies, LVMH, and BNP Paribas. The Banque de France published its "
        "quarterly projection while the INSEE national statistics institute released "
        "updated employment data. The Ministry of Economy, Finance, and Industrial and "
        "Digital Sovereignty coordinated with the French Treasury (Direction générale du "
        "Trésor). Le Monde and Les Échos reported on statements by the CEO of Airbus "
        "and the president of the French Academy of Sciences. The French Democratic "
        "Confederation of Labour (CFDT) and the General Confederation of Labour (CGT) "
        "debated pension-system impacts. Sanofi, Renault Group, and Société Générale "
        "outlined digital transformation plans. The French National Centre for "
        "Scientific Research (CNRS) and École Polytechnique presented AI policy "
        "research. Bpifrance and the French Tech initiative discussed start-up ecosystem "
        "competitiveness."
    ),
    "DE": (
        "The Federation of German Industries (BDI) held an extraordinary session in "
        "Berlin with board members of SAP SE, Siemens AG, and Deutsche Telekom AG. The "
        "Deutsche Bundesbank adjusted its growth forecast while the German Council of "
        "Economic Experts (Sachverständigenrat) published its annual report. Federal "
        "Minister of Economics Robert Habeck's team at the Federal Ministry for Economic "
        "Affairs and Climate Action (BMWK) coordinated with the German Chambers of "
        "Commerce and Industry (DIHK). Handelsblatt and Der Spiegel covered statements "
        "by the CTO of Deutsche Bank AG and the president of the Fraunhofer-Gesellschaft. "
        "IG Metall and the German Trade Union Confederation (DGB) discussed workforce "
        "transition impacts. Bitkom e.V. released its annual digital-economy monitor. "
        "Volkswagen AG, BASF SE, and Allianz SE outlined their strategic responses. The "
        "Max Planck Society, the Leibniz Association, and the Helmholtz Association "
        "announced joint research clusters. The German Federal Employment Agency "
        "(Bundesagentur für Arbeit) and KfW Development Bank published regional impact "
        "assessments."
    ),
    "GR": (
        "The Hellenic Federation of Enterprises (SEV) organised a summit in Athens with "
        "executives from Hellenic Petroleum (HELLENiQ Energy), OTE Group (Cosmote), and "
        "National Bank of Greece. The Bank of Greece published its monetary policy report "
        "while the Centre of Planning and Economic Research (KEPE) released GDP "
        "projections. The Ministry of National Economy and Finance coordinated with the "
        "Athens Chamber of Commerce and Industry (ACCI). Kathimerini and Naftemporiki "
        "covered statements by the CEO of Piraeus Bank and the rector of the National "
        "Technical University of Athens (NTUA). The General Confederation of Greek "
        "Workers (GSEE) raised concerns about youth unemployment. Eurobank, Alpha Bank, "
        "and Motor Oil Hellas discussed investment plans. The Foundation for Economic & "
        "Industrial Research (IOBE) and the Hellenic Foundation for European and Foreign "
        "Policy (ELIAMEP) presented structural reform analyses. Enterprise Greece and "
        "the Hellenic Development Bank announced export-support programmes."
    ),
    "HU": (
        "The Confederation of Hungarian Employers and Industrialists (MGYOSZ) met in "
        "Budapest with MOL Group, OTP Bank, and Richter Gedeon. The Magyar Nemzeti Bank "
        "(MNB) published inflation targets while the Institute for Economic and "
        "Enterprise Research (GVI) at the Budapest Chamber of Commerce assessed FDI "
        "trends. The Ministry for National Economy coordinated with the Hungarian "
        "Chamber of Commerce and Industry (MKIK). HVG and Portfolio.hu reported on "
        "statements by the CEO of Magyar Telekom and the president of the Hungarian "
        "Academy of Sciences (MTA). The National Federation of Workers' Councils "
        "(Munkástanácsok) and the Trade Union of Commercial Employees (KASZ) discussed "
        "wage policies. Wizz Air, BorsodChem (Wanhua), and CIG Pannónia outlined "
        "expansion strategies. The Budapest University of Technology and Economics (BME) "
        "and the Centre for Economic and Regional Studies (KRTK) published research on "
        "regional convergence. The Hungarian Development Bank (MFB) announced EU-funded "
        "infrastructure projects."
    ),
    "IE": (
        "IBEC, Ireland's largest employer body, held a policy summit in Dublin with "
        "representatives from CRH plc, Ryanair Holdings, and AIB Group. The Central "
        "Bank of Ireland published its financial stability review while the Economic and "
        "Social Research Institute (ESRI) released updated fiscal projections. The "
        "Department of Enterprise, Trade and Employment coordinated with IDA Ireland and "
        "Enterprise Ireland. The Irish Times and the Irish Independent reported on "
        "statements by the CEO of Kerry Group and the provost of Trinity College Dublin. "
        "The Irish Congress of Trade Unions (ICTU) raised concerns about housing costs "
        "impacting labour mobility. Smurfit Kappa, Kingspan Group, and Bank of Ireland "
        "discussed capital expenditure plans. Science Foundation Ireland (SFI) and the "
        "Tyndall National Institute presented semiconductor research findings. The "
        "National Treasury Management Agency (NTMA) and the Ireland Strategic Investment "
        "Fund (ISIF) announced green-bond initiatives."
    ),
    "IT": (
        "Confindustria convened an emergency session in Rome with executives from Enel "
        "SpA, Eni SpA, and Intesa Sanpaolo. The Banca d'Italia revised its growth "
        "outlook while ISTAT published updated employment statistics. The Ministry of "
        "Economy and Finance (MEF) coordinated with the Italian Trade Agency (ICE) and "
        "Cassa Depositi e Prestiti (CDP). Il Sole 24 Ore and Corriere della Sera "
        "reported on statements by the CEO of UniCredit and the rector of Politecnico di "
        "Milano. The Italian General Confederation of Labour (CGIL) and CISL debated "
        "industrial policy reform. Ferrari NV, Leonardo SpA, and Generali Group outlined "
        "strategic investments. The Italian National Research Council (CNR) and the "
        "Fondazione Bruno Kessler presented AI governance research. Mediobanca and the "
        "Italian Banking Association (ABI) published a lending survey. SACE SpA "
        "announced new export credit instruments."
    ),
    "LV": (
        "The Employers' Confederation of Latvia (LDDK) held discussions in Riga with "
        "airBaltic, Latvenergo, and Latvijas Finieris. The Bank of Latvia (Latvijas "
        "Banka) published an economic review while the Latvian Council of Science "
        "assessed R&D investment levels. The Ministry of Economics coordinated with the "
        "Latvian Chamber of Commerce and Industry (LTRK) and the Investment and "
        "Development Agency of Latvia (LIAA). Delfi Latvia and Diena reported on "
        "statements by the CEO of Tet (Lattelecom) and the rector of the University of "
        "Latvia. The Free Trade Union Confederation of Latvia (LBAS) discussed workforce "
        "emigration trends. Mikrotīkls (MikroTik), Printful, and Citadele Bank outlined "
        "growth plans. The Institute of Electronics and Computer Science (EDI) and Riga "
        "Technical University presented digital-transformation research. The "
        "cross-border Rail Baltica project office and the Freeport of Riga Authority "
        "commented on logistics infrastructure."
    ),
    "LT": (
        "The Lithuanian Confederation of Industrialists (LPK) met in Vilnius with "
        "representatives from Ignitis Group, Girteka Logistics, and Maxima Group. The "
        "Bank of Lithuania (Lietuvos bankas) released its financial stability review "
        "while the Lithuanian Free Market Institute (LFMI) published a regulatory burden "
        "study. The Ministry of the Economy and Innovation coordinated with Enterprise "
        "Lithuania and Invest Lithuania. Delfi Lithuania and Verslo žinios reported on "
        "statements by the CEO of Telia Lietuva and the rector of Vilnius University. "
        "The Lithuanian Trade Union Confederation (LPSK) raised wage-competitiveness "
        "concerns. Vinted, Tesonet, and Šiaulių bankas discussed fintech ecosystem "
        "growth. Vilnius Tech and Kaunas University of Technology (KTU) presented "
        "advanced-manufacturing research. The Lithuanian Centre for Social Sciences and "
        "the Research Council of Lithuania announced joint EU-funded projects."
    ),
    "LU": (
        "The Luxembourg Business Federation (UEL) held consultations in Luxembourg City "
        "with ArcelorMittal, SES SA, and Banque Internationale à Luxembourg (BIL). The "
        "Banque centrale du Luxembourg (BCL) published its economic projections while "
        "STATEC released updated GDP figures. The Ministry of the Economy coordinated "
        "with Luxinnovation and the Luxembourg Chamber of Commerce. Luxemburger Wort and "
        "Paperjam reported on statements by the CEO of Cactus Group and the rector of "
        "the University of Luxembourg. The Luxembourg Confederation of Independent Trade "
        "Unions (OGBL) discussed cross-border commuter policies. Eurofins Scientific, "
        "RTL Group, and the European Investment Bank (EIB) — headquartered in Luxembourg "
        "— outlined ESG investment strategies. The Luxembourg Institute of Science and "
        "Technology (LIST) and the Luxembourg Institute of Health (LIH) presented "
        "research findings. The Commission de Surveillance du Secteur Financier (CSSF) "
        "issued new fintech guidelines."
    ),
    "MT": (
        "The Malta Chamber of Commerce, Enterprise and Industry met in Valletta with "
        "representatives from Bank of Valletta, GO plc, and Malta International Airport. "
        "The Central Bank of Malta published its economic update while the Malta Council "
        "for Economic and Social Development (MCESD) assessed labour-market trends. The "
        "Ministry for the Economy, European Funds and Lands coordinated with Malta "
        "Enterprise and the Malta Financial Services Authority (MFSA). The Times of "
        "Malta and MaltaToday reported on the CEO of Enemalta and the rector of the "
        "University of Malta. The General Workers' Union (GWU) raised housing-cost "
        "concerns for foreign workers. Tipico, Betsson Group (Malta operations), and "
        "HSBC Malta discussed the iGaming sector's contribution. The Malta Information "
        "Technology Agency (MITA) and the Malta College of Arts, Science and Technology "
        "(MCAST) presented digital-skills research. Transport Malta and the Malta "
        "Freeport Authority commented on supply-chain resilience."
    ),
    "NL": (
        "VNO-NCW, the Confederation of Netherlands Industry and Employers, held its "
        "annual meeting in The Hague with executives from Royal Dutch Shell (Shell plc), "
        "ASML, and ING Group. De Nederlandsche Bank (DNB) published its financial "
        "stability overview while the CPB Netherlands Bureau for Economic Policy "
        "Analysis released updated forecasts. The Ministry of Economic Affairs and "
        "Climate Policy coordinated with the Netherlands Enterprise Agency (RVO). Het "
        "Financieele Dagblad and NRC Handelsblad reported on statements by the CEO of "
        "Philips and the president of the Royal Netherlands Academy of Arts and Sciences "
        "(KNAW). The Federation of Dutch Trade Unions (FNV) discussed collective "
        "bargaining trends. Unilever, Heineken, and ABN AMRO Bank outlined investment "
        "plans. TNO (Netherlands Organisation for Applied Scientific Research) and Delft "
        "University of Technology presented semiconductor-ecosystem research. The Dutch "
        "Authority for the Financial Markets (AFM) and Invest-NL announced green-finance "
        "frameworks."
    ),
    "PL": (
        "The Polish Confederation Lewiatan held a summit in Warsaw with representatives "
        "from PKN Orlen, PZU Group, and KGHM Polska Miedź. The National Bank of Poland "
        "(NBP) released its inflation report while the Polish Economic Institute (PIE) "
        "published an industrial-transformation study. The Ministry of Development and "
        "Technology coordinated with the Polish Investment and Trade Agency (PAIH) and "
        "the Polish Development Fund (PFR). Rzeczpospolita and Gazeta Wyborcza reported "
        "on statements by the CEO of PKO Bank Polski and the rector of the University of "
        "Warsaw. The Independent Self-Governing Trade Union Solidarity (NSZZ Solidarność) "
        "discussed energy-transition workforce impacts. CD Projekt, Allegro, and Bank "
        "Pekao outlined technology investment strategies. The Polish Academy of Sciences "
        "(PAN) and the Warsaw University of Technology presented joint research on "
        "electromobility. The National Centre for Research and Development (NCBR) "
        "announced EU-funded innovation programmes."
    ),
    "PT": (
        "The Confederation of Portuguese Industry (CIP) met in Lisbon with executives "
        "from Galp Energia, EDP (Energias de Portugal), and Jerónimo Martins. The Banco "
        "de Portugal published its economic bulletin while the Foundation for Science and "
        "Technology (FCT) assessed R&D spending levels. The Ministry of Economy "
        "coordinated with AICEP Portugal Global and the Portuguese Chamber of Commerce "
        "and Industry (CCIP). Jornal de Negócios and Expresso reported on the CEO of "
        "Sonae and the rector of the University of Lisbon. The General Confederation of "
        "Portuguese Workers (CGTP-IN) raised productivity-gap concerns. The Navigator "
        "Company, Mota-Engil, and Millennium BCP discussed infrastructure projects. The "
        "University of Porto's Faculty of Engineering and the Instituto Superior Técnico "
        "(IST) presented renewable-energy research. IAPMEI (Agency for Competitiveness "
        "and Innovation) and Startup Portugal announced scale-up support measures."
    ),
    "RO": (
        "The Alliance of Romanian Employers' Confederations (ACPR) convened in Bucharest "
        "with representatives from OMV Petrom, Banca Transilvania, and Electrica SA. The "
        "National Bank of Romania (BNR) released its inflation report while the National "
        "Commission for Strategy and Prognosis (CNSP) updated economic forecasts. The "
        "Ministry of Economy coordinated with the Romanian Chamber of Commerce and "
        "Industry (CCIR) and Invest Romania. Ziarul Financiar and Economica.net reported "
        "on statements by the CEO of Romgaz and the rector of the University of "
        "Bucharest. The National Trade Union Bloc (BNS) and Cartel ALFA discussed "
        "minimum-wage indexation. UiPath, Bitdefender, and Digi Communications outlined "
        "tech-sector growth. The Romanian Academy and the Polytechnic University of "
        "Bucharest presented digital-infrastructure research. EximBank Romania and the "
        "Romanian Development Bank (BERD partnerships) announced SME credit facilities."
    ),
    "SK": (
        "The National Union of Employers (RÚZ) met in Bratislava with executives from "
        "Slovenský plynárenský priemysel (SPP), Slovnaft, and Tatra banka. The National "
        "Bank of Slovakia (NBS) published its macroeconomic forecast while the Slovak "
        "Academy of Sciences (SAV) released a demographic trends study. The Ministry of "
        "Economy coordinated with the Slovak Investment and Trade Development Agency "
        "(SARIO) and the Slovak Chamber of Commerce and Industry (SOPK). SME (Sme.sk) "
        "and Hospodárske noviny reported on statements by the CEO of Slovak Telekom and "
        "the rector of Comenius University. The Confederation of Trade Unions (KOZ SR) "
        "discussed automotive-sector wage trends. Eset, Asseco Central Europe, and "
        "Slovenská sporiteľňa outlined digital strategy. The Slovak University of "
        "Technology in Bratislava (STU) and the Technical University of Košice (TUKE) "
        "presented advanced-materials research. The Slovak Innovation and Energy Agency "
        "(SIEA) announced green-transition funding."
    ),
    "SI": (
        "The Chamber of Commerce and Industry of Slovenia (GZS) hosted a forum in "
        "Ljubljana with Krka dd, Petrol dd, and NLB Group. The Bank of Slovenia (Banka "
        "Slovenije) released its financial stability review while the Institute for "
        "Macroeconomic Analysis and Development (IMAD/UMAR) updated GDP projections. The "
        "Ministry of the Economy, Tourism and Sport coordinated with the SPIRIT Slovenia "
        "public agency. Delo and Finance reported on statements by the CEO of Lek (a "
        "Sandoz/Novartis company) and the rector of the University of Ljubljana. The "
        "Association of Free Trade Unions of Slovenia (ZSSS) discussed productivity "
        "benchmarks. Outfit7, Bitstamp, and Triglav Group outlined technology investment. "
        "The Jožef Stefan Institute and the National Institute of Chemistry presented "
        "materials-science research. The Slovenian Research and Innovation Agency (ARIS) "
        "and SID Bank announced EU co-funded innovation grants."
    ),
    "ES": (
        "The Spanish Confederation of Employers' Organizations (CEOE) held a plenary "
        "session in Madrid with executives from Iberdrola, Banco Santander, and "
        "Telefónica. The Banco de España published its quarterly economic bulletin while "
        "the Centre for Economic Forecasting (CEPREDE) at Universidad Autónoma de Madrid "
        "released employment projections. The Ministry of Economy, Trade and Enterprise "
        "coordinated with ICEX Spain Trade & Investment and CDTI (Centre for Industrial "
        "Technological Development). El País and Expansión reported on statements by the "
        "CEO of Inditex and the president of the Spanish National Research Council "
        "(CSIC). The Workers' Commissions (CCOO) and the General Union of Workers (UGT) "
        "debated labour reform outcomes. Repsol, CaixaBank, and Amadeus IT Group "
        "outlined digital transformation roadmaps. The Barcelona Supercomputing Center "
        "(BSC) and the IESE Business School presented competitiveness analyses. The "
        "Official Credit Institute (ICO) and COFIDES announced export financing "
        "programmes."
    ),
    "SE": (
        "The Confederation of Swedish Enterprise (Svenskt Näringsliv) held its annual "
        "conference in Stockholm with leaders from Ericsson, Volvo Group, and H&M Group. "
        "Sveriges Riksbank published its monetary policy report while the National "
        "Institute of Economic Research (Konjunkturinstitutet) released labour-market "
        "projections. The Ministry of Finance coordinated with Business Sweden and the "
        "Swedish Agency for Economic and Regional Growth (Tillväxtverket). Dagens "
        "Industri and Svenska Dagbladet reported on statements by the CEO of Spotify and "
        "the president of the Royal Swedish Academy of Sciences. The Swedish Trade Union "
        "Confederation (LO) and the Swedish Confederation of Professional Employees "
        "(TCO) discussed green-transition skills gaps. Atlas Copco, Sandvik, and "
        "Handelsbanken outlined investment strategies. The KTH Royal Institute of "
        "Technology and the Karolinska Institute presented research on life-science "
        "innovation. Vinnova (Sweden's innovation agency) and the Wallenberg Foundations "
        "announced AI research funding."
    ),
}

# Fallback for countries without a dedicated narrative
_DEFAULT_ENTITY_NARRATIVE = ""


def build_seed_document(country_code: str, scenario: str = "") -> str:
    """Generate a seed document for a country that feeds MiroFish ontology generation.

    The user-supplied *scenario* is the primary simulation requirement.
    Country profile data is appended as supplementary context so MiroFish
    can ground its ontology in real indicators without overriding the
    user's intent.
    """
    country = _COUNTRY_BY_CODE.get(country_code.upper())
    if country is None:
        raise ValueError(f"Unknown country code: {country_code}")

    # --- Primary: user scenario drives the simulation -----------------
    lines: list[str] = []
    if scenario:
        lines.append(f"Simulation Requirement: {scenario}")
        lines.append("")

    # --- Supplementary: country reference data ------------------------
    lines.extend([
        f"Country context for {country['name']} ({country['code']}):",
        f"  Capital: {country['capital']}",
        f"  Population: {country['population']:,}",
        f"  GDP per capita (EUR): {country['gdp_per_capita']:,}",
        f"  Unemployment rate: {country['unemployment_rate']}%",
        f"  EU member state.",
    ])

    # --- Named entities for NER / knowledge-graph seeding -------------
    entity_narrative = _COUNTRY_ENTITIES.get(
        country_code.upper(), _DEFAULT_ENTITY_NARRATIVE
    )
    if entity_narrative:
        lines.append("")
        lines.append(f"Stakeholder landscape in {country['name']}:")
        lines.append(entity_narrative)

    # --- Closing instruction ------------------------------------------
    lines.append("")
    lines.append(
        "Use the Eurostat baseline indicators and the named stakeholders "
        "above as reference data. The simulation MUST focus on the "
        "requirement stated at the top of this document."
    )
    return "\n".join(lines)


# ======================================================================
# In-memory job store
# ======================================================================

class JobStore:
    """Thread-safe in-memory store for simulation jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self, country_code: str, scenario: str, params: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "country_code": country_code.upper(),
            "scenario": scenario,
            "model": params.get("model"),
            "params": params,
            "status": "queued",
            "stage": None,
            "stages_completed": [],
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[dict]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()


store = JobStore()
mf = MiroFishClient()

# ======================================================================
# Pipeline orchestration (runs in background thread)
# ======================================================================

PIPELINE_STAGES = [
    "ontology_generate",
    "graph_build",
    "simulation_create",
    "simulation_prepare",
    "generate_profiles",
    "simulation_start",
    "simulation_run",
    "report_generate",
]


def _run_pipeline(job_id: str) -> None:
    """Execute all MiroFish stages sequentially for a job."""
    job = store.get(job_id)
    if job is None:
        return

    country_code = job["country_code"]
    scenario = job["scenario"]
    params = job["params"]
    seed_doc = build_seed_document(country_code, scenario)

    store.update(job_id, status="running", stage="ontology_generate", progress=0)

    try:
        # Stage 1 – Ontology generation (multipart/form-data)
        country = _COUNTRY_BY_CODE[country_code]
        project_name = params.get("project_name", f"MiroFish – {country['name']}")
        ontology_result = mf.generate_ontology(
            simulation_requirement=seed_doc,
            project_name=project_name,
        )
        ontology_data = ontology_result.get("data", {}) if isinstance(ontology_result, dict) else {}
        project_id = ontology_data.get("project_id")
        store.update(
            job_id,
            stage="graph_build",
            stages_completed=["ontology_generate"],
            progress=12,
        )

        # Stage 2 – Graph build (needs project_id, async — poll task until done)
        graph_result = mf.build_graph(project_id, **params.get("graph", {}))
        graph_data = graph_result.get("data", {}) if isinstance(graph_result, dict) else {}
        graph_task_id = graph_data.get("task_id")
        if graph_task_id:
            mf.wait_for_task(graph_task_id, poll_interval=5.0, timeout=LLM_TIMEOUT)
        store.update(
            job_id,
            stage="simulation_create",
            stages_completed=["ontology_generate", "graph_build"],
            progress=25,
        )

        # Stage 3 – Create simulation (needs project_id, returns simulation_id + graph_id)
        sim_result = mf.create_simulation(project_id, **params.get("simulation", {}))
        sim_data = sim_result.get("data", {}) if isinstance(sim_result, dict) else {}
        sim_id = sim_data.get("simulation_id")
        graph_id = sim_data.get("graph_id")
        store.update(
            job_id,
            stage="simulation_prepare",
            stages_completed=["ontology_generate", "graph_build", "simulation_create"],
            progress=37,
        )

        # Stage 4 – Prepare simulation (needs simulation_id, async — poll until ready)
        prepare_result = mf.prepare_simulation(sim_id, **params.get("prepare", {}))
        prepare_data = prepare_result.get("data", {}) if isinstance(prepare_result, dict) else {}
        prepare_task_id = prepare_data.get("task_id")
        if not prepare_data.get("already_prepared"):
            mf.wait_for_prepare(sim_id, prepare_task_id, poll_interval=5.0, timeout=LLM_TIMEOUT)
        store.update(
            job_id,
            stage="generate_profiles",
            stages_completed=["ontology_generate", "graph_build", "simulation_create", "simulation_prepare"],
            progress=50,
        )

        # Stage 5 – Generate profiles (needs graph_id)
        profile_params = {"parallel_profile_count": 8, **params.get("profiles", {})}
        profiles_result = mf.generate_profiles(graph_id, **profile_params)
        store.update(
            job_id,
            stage="simulation_start",
            stages_completed=[
                "ontology_generate", "graph_build", "simulation_create",
                "simulation_prepare", "generate_profiles",
            ],
            progress=62,
        )

        # Stage 6 – Start simulation (needs simulation_id)
        start_result = mf.start_simulation(sim_id, **params.get("start", {}))
        store.update(
            job_id,
            stage="simulation_run",
            stages_completed=[
                "ontology_generate", "graph_build", "simulation_create",
                "simulation_prepare", "generate_profiles", "simulation_start",
            ],
            progress=70,
        )

        # Stage 7 – Wait for simulation to finish (polls run-status)
        # Simulation runs can take hours depending on agent count and rounds;
        # use a separate, much longer timeout than the per-LLM-call timeout.
        run_result = mf.wait_for_simulation(sim_id, poll_interval=10.0, timeout=SIM_RUN_TIMEOUT)
        store.update(
            job_id,
            stage="report_generate",
            stages_completed=[
                "ontology_generate", "graph_build", "simulation_create",
                "simulation_prepare", "generate_profiles", "simulation_start",
                "simulation_run",
            ],
            progress=85,
        )

        # Stage 8 – Generate report (needs simulation_id, async — poll task)
        report_result = mf.generate_report(sim_id, **params.get("report", {}))
        report_data = report_result.get("data", {}) if isinstance(report_result, dict) else {}
        report_task_id = report_data.get("task_id")
        if report_task_id:
            mf.wait_for_task(report_task_id, poll_interval=5.0, timeout=LLM_TIMEOUT)

        store.update(
            job_id,
            status="completed",
            stage=None,
            stages_completed=PIPELINE_STAGES,
            progress=100,
            result={
                "project_id": project_id,
                "graph_id": graph_id,
                "simulation_id": sim_id,
                "ontology": ontology_result,
                "graph": graph_result,
                "simulation": sim_result,
                "profiles": profiles_result,
                "run": run_result,
                "report": report_result,
            },
        )

    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))


# ======================================================================
# Flask application
# ======================================================================

app = Flask(__name__)
CORS(app)


# --- Health ---------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    """Proxy health-check: checks both this server and MiroFish."""
    mirofish_ok = False
    try:
        mf.health()
        mirofish_ok = True
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "mirofish_reachable": mirofish_ok,
        "mirofish_url": MIROFISH_BASE_URL,
    })


# --- Static data ----------------------------------------------------------

@app.route("/api/countries", methods=["GET"])
def api_countries():
    return jsonify(EU_COUNTRIES)


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    jobs = store.list_all()
    return jsonify(jobs)


# --- Simulate -------------------------------------------------------------

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Start a full simulation pipeline for a given EU country.

    Expected JSON body:
        {
            "country_code": "DE",
            "scenario": "optional free-text scenario description",
            "params": { ... optional per-stage overrides ... }
        }
    """
    body = request.get_json(silent=True) or {}
    country_code = body.get("country_code", "").upper()

    if country_code not in _COUNTRY_BY_CODE:
        return jsonify({"error": f"Unknown country code: {country_code}"}), 400

    scenario = body.get("scenario", "")
    params = body.get("params", {})
    model = body.get("model")
    if model:
        params = {**params, "model": model}

    job = store.create(country_code, scenario, params)

    thread = threading.Thread(target=_run_pipeline, args=(job["id"],), daemon=True)
    thread.start()

    return jsonify({"job_id": job["id"], "status": "queued"}), 202


# --- Status / Results -----------------------------------------------------

@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "stages_completed": job["stages_completed"],
        "progress": job["progress"],
        "model": job.get("model"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job["error"],
    })


@app.route("/api/results/<job_id>", methods=["GET"])
def api_results(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({
            "error": "Results not ready",
            "status": job["status"],
            "progress": job["progress"],
        }), 409

    return jsonify(job["result"])


# --- Chat -----------------------------------------------------------------

@app.route("/api/chat/<job_id>", methods=["POST"])
def api_chat(job_id: str):
    """Chat about a completed simulation's report.

    Expected JSON body:
        { "message": "user question about the report" }
    """
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Simulation not completed yet"}), 409

    body = request.get_json(silent=True) or {}
    message = body.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    sim_id = job["result"]["simulation_id"]
    chat_history = body.get("chat_history", [])
    try:
        reply = mf.report_chat(sim_id, message, chat_history)
    except MiroFishAPIError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"reply": reply})


# ======================================================================
# Entry point
# ======================================================================


@app.route("/api/health/deep", methods=["GET"])
def api_health_deep():
    """Detailed health check for showcase readiness."""
    try:
        mf.health()
        mirofish_ok = True
    except Exception:
        mirofish_ok = False

    try:
        ollama_resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3.0)
        ollama_ok = ollama_resp.ok
        tags = ollama_resp.json().get("models", []) if ollama_ok else []
        model_names = {m.get("name") for m in tags if isinstance(m, dict)}
        model_ready = not model_names or DEFAULT_MODEL in model_names
    except Exception:
        ollama_ok = False
        model_ready = False
        model_names = set()

    ready = mirofish_ok and ollama_ok and model_ready
    return jsonify({
        "status": "ok" if ready else "degraded",
        "mirofish_reachable": mirofish_ok,
        "mirofish_url": MIROFISH_BASE_URL,
        "ollama_reachable": ollama_ok,
        "model_ready": model_ready,
        "model": DEFAULT_MODEL,
        "available_models": sorted(model_names),
    }), 200 if ready else 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
