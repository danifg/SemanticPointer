# SemanticPointer
This repository includes the code of the Semantic Dependency Parser with Pointer Networks described in ACL paper [Transition-based Semantic Dependency Parsing with Pointer Networks](https://www.aclweb.org/anthology/2020.acl-main.629.pdf). The implementation is based on the dependency parser by Ma et al. (2018) (https://github.com/XuezheMax/NeuroNLP2) and reuses part of its code.

### Requirements
This implementation requires Python 2.7, PyTorch 0.3.1 and Gensim >= 0.12.0.
  

### Experiments
First of all, you need to include official datasets in SDP format in the ``data`` folder, and use the following script to convert them to the proper input DAG format. For instance, for the DM formalism:

     python ./scripts/convert.py ./data/en_dm_train.sdp ./data/en_dm_train.dag
     python ./scripts/convert.py ./data/en_dm_dev.sdp ./data/en_dm_dev.dag
     python ./scripts/convert.py ./data/en_dm_test.id.sdp ./data/en_dm_test.id.dag
     python ./scripts/convert.py ./data/en_dm_test.ood.sdp ./data/en_dm_test.ood.dag
	
To train the parser, you need to include the pre-trained word embeddings in the ``embs`` folder and run the following script, indicating the model name and the formalism that you want to use:

    ./scripts/run_parser.sh <model_name> <dm|pas|psd>


Finally, to evaluate the best trained model on the test set, just use the oficial script to compute the Labelled and Unlabelled F1 scores (you must indicate the epoch of the best reported model on the development set, the chosen formalism and the trained model name):

    ./scripts/eval.sh <best_epoch> <dm|pas|psd> <model_name>


### Citation

	@inproceedings{fernandez-gonzalez-gomez-rodriguez-2020-transition,
    title = "Transition-based Semantic Dependency Parsing with Pointer Networks",
    author = "Fern{\'a}ndez-Gonz{\'a}lez, Daniel  and
      G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.629",
    pages = "7035--7046"}
    
### Acknowledgments
This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150), from the ANSWER-ASAP project (TIN2017-85160-C2-1-R) from MINECO, and from Xunta de Galicia (ED431B 2017/01, ED431G 2019/01).

### Contact
If you have any suggestion, inquiry or bug to report, please contact d.fgonzalez@udc.es.
