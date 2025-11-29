# Neuromorphic Bird Classifier Data and Norse model

<p align="center">
	<img src="https://img.shields.io/github/license/Neuromorphicism/neuromorphic-bird-classifier-data-and-norse-model" />
	<a href="https://github.com/Neuromorphicism/neuromorphic-bird-classifier-data-and-norse-model/pulse" alt="Activity">
        <img src="https://img.shields.io/github/last-commit/Neuromorphicism/neuromorphic-bird-classifier-data-and-norse-model" />
    </a>
    <a href="https://open-neuromorphic.org/neuromorphic-computing/">
	    <img src="https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue">
	</a>
</p>

Data and Norse model for SNN bird classification

<br>

## Data Authors
Frank Vincentz, Aviceda, Alice Stock Footages, Nederlands Instituut voor Beeld en Geluid, Natuur Digitaal (Marc Plomp), The Nature Box, Badarin, Bureau of Land Management Oregon and Washington, Devra, Paul Danese, Chris Light.

Their long videos in WEBM format were transformed into short event videos in DAT format. If an author is not stated then the sourced work was in the public domain. None of the data nor libraries used in this article are in the paid commercial domain. The listed authors data is shared under the Creative Commons Attribution-Share Alike 3.0 license.

<br>

## Train the SNN Norse Model on Linux

AEStream currently works only on Linux and that is why this Norse Model can only be trained on that OS.

Open this repository folder in a terminal and run:

```bash
pip install -r requirements.txt
python ./norse-model-app/norse-train.py
```

It is even better to use Python 3.8+ and conda:

```bash
conda 
conda activate
pip install -r requirements.txt
python ./norse-model-app/norse-train.py
```

The trained model will be placed in `saved-snn-models/model.pth` file.

You can use this model in NeuroBCDA: https://github.com/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui

<br>

## Add new webm data and transform it into events

```bash
cd events-from-video-app/IEBCS-main
python ./examples/00_video_to_events/1_example_video_to_events.py
```

The created file in the DAT format is now in the `examples/00_video_to_events/outputs` folder. Move that file to the `event-birds/` folder according to the name of a bird then run the SNN training again to get a better model.
