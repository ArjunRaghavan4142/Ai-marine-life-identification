"""Download the 481-class fish species dataset from Roboflow."""
from roboflow import Roboflow
import pathlib

DEST = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "fish_species"

rf = Roboflow(api_key="Bh5TAIsYQ1L56EOsv4jP")
project = rf.workspace("fish-classifiers").project("fish-species-classification-mdfri")
versions = project.versions()
for v in versions:
    print(f"Version: {v.version}")
version = project.version(versions[0].version)
dataset = version.download("yolov8", location=str(DEST))
print(f"Downloaded to {DEST}")
