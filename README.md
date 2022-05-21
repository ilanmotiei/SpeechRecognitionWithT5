# SpeechRecognitionWithT5

For setup:

```
git clone https://github.com/ilanmotiei/SpeechRecognitionWithT5.git
cd SpeechRecognitionWithT5
pip3 install -r requirements.txt
```

Download the available checkpoint from here: 
https://drive.google.com/file/d/1sCWHBxowRvczVzeUx6Ke_aYNQK0O8yPv/view?usp=sharing

For checking the results on the test-clean subset of Libri-Speech, download its pre-processed version from here, and unzip:
https://drive.google.com/file/d/1ZoiUUkrbAo2nDHocaP_o38GfEn2teZWU/view?usp=sharing

Then, run:
```
python3 src/eval.py
```

For getting the results on that subset.
Make sure you have an available CUDA gpu on your machine.
