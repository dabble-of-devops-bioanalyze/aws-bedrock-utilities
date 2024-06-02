import os

import pandas as pd
import funcy
import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

datasets = [
    "associationByDatasourceDirect",
    "associationByDatasourceIndirect",
    "associationByDatatypeDirect",
    "associationByDatatypeIndirect",
    "associationByOverallDirect",
    "associationByOverallIndirect",
    "baselineExpression",
    "diseaseToPhenotype",
    "diseases",
    "drugWarnings",
    "ebisearchAssociations",
    "ebisearchEvidence",
    "errors",
    "evidence",
    "fda",
    "go",
    "hpo",
    "indication",
    "interaction",
    "interactionEvidence",
    "knownDrugsAggregated",
    "literature",
    "mechanismOfAction",
    "molecule",
    "mousePhenotypes",
    "pharmacogenomics",
    "reactome",
    "searchDisease",
    "searchDrug",
    "searchTarget",
    "targetEssentiality",
    "targetPrioritisation",
    "targets",
]


def main(version: str = "24.03"):
    for ds in datasets:
        c = f"rsync - rpltvz - -delete rsync.ebi.ac.uk::pub/databases/opentargets/platform/{version}/output/etl/parquet/{ds} ."
        os.system(c)
