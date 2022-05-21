# SpeechRecognitionWithT5

For setup:

```
git clone https://github.com/ilanmotiei/SpeechRecognitionWithT5.git
cd SpeechRecognitionWithT5
pip3 install -r requirements.txt
```

Download the available checkpoint from here: 
https://drive.google.com/file/d/1sCWHBxowRvczVzeUx6Ke_aYNQK0O8yPv/view?usp=sharing

For checking the results on the test subsets of Libri-Speech, download the pre-processed versions from here, and unzip them:
test-other: https://drive.google.com/file/d/1eclUQvae94-Xobj0pBCglMVdgoPEnF60/view?usp=sharing
test-clean: https://drive.google.com/file/d/1WTFTvyzekYVMm_80UlQSXS8dCvHZeNgU/view?usp=sharing

Place both those items in the root directory of the cloned repository.

Then, run:
```
python3 src/eval.py
```

And you'll be getting the results on that subset.
Make sure you have an available CUDA gpu on your machine.
