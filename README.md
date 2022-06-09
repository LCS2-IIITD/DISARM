# DISARM
Repository for "DISARM: Detecting the Victims Targeted by Harmful Memes" @ NAACL'22 (Findings)

arXiv'ed version: https://arxiv.org/abs/2205.05738

The enclosure has the following (besides a readme.txt):
<ol>
<li> requirements.txt
<li> LightWt_basic_model_BERT_Supplementary.ipynb
<li> LightWt_basic_model_BERT_Supplementary.py
</ol>

This implementation also requires downloading of CLIP model (as follows):<br>

<code>MODELS = {</code><br>
     <code>"RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",</code><br>
     <code>"RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",</code><br>
     <code>"RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",</code><br>
     <code>"ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",    </code><br>
}</code><br>
<code>! wget {MODELS["ViT-B/32"]} -O clip_model.pt</code>

The enclosed .py file contains all the code in the pipeline in sequence.
It can be preferrably run by placing blocks sequentially in a notebook.
Or the notebook itself can be ran.
