

# 🤖🔜👶❓ Benchmarking Progress to Infant-Level Physical Reasoning in AI  
    
This repository is used to track the release of code and data corresponding to the paper **[Benchmarking Progress to Infant-Level Physical Reasoning in AI](https://openreview.net/forum?id=9NjqD9i48M)** accepted for publication at the _Transactions on Machine Learning Research_ (TMLR). Code and data and being cleaned an prepared for release, this should occur within the next few weeks. For anyone looking to get started ahead of this release, you can find (largely undocumented and anonoymized for double-blind review) code and data we used to generate our results [here](https://openreview.net/forum?id=9NjqD9i48M&noteId=jl0ICHG5k6).

#### Quick links
* [Website](http://inflevel.allenai.org/)
* [Paper PDF](https://openreview.net/pdf?id=9NjqD9i48M)
* [Reviews on OpenReview](https://openreview.net/forum?id=9NjqD9i48M)
* <a href="#-citation">📜 Citation</a>
  
## 📝 What is the InfLevel Benchmark?  
  
We introduce the, _evaluation only_, Infant-Level Physical Reasoning Benchmark (InfLevel) to study a fundamental question: **to what extent do modern AI systems comprehend the physical world?**  
  
We propose to evaluate existing AI systems using the violation-of-expectations (VoE) paradigm used by developmental psychologists when studying infants. In this paradigm, infants' are typically presented with a physically implausible event, which violates a physical principle under investigation, and a physically plausible event, which conforms to the principle. With appropriate controls, greater surprise at the implausible event, with looking time as the canonical surrogate measure quantifying surprise, is taken to indicate that the child brought to bear the relevant principle to form expectations about how the events would unfold and subsequently detected the violation in the physically implausible event. Using this experimental technique, researchers have shown that, by 4.5 months of age, infants correctly reason about objects' displacements and interactions in many different physical events. How do AI systems compare?  
  
Our InfLevel benchmark focuses on three physical principles that have been extensively studied, Continuity, Solidity, and Gravity, modeled on past experimental work in developmental psychology. See below for some examples of videos in InfLevel.  
  
### Continuity
<img src="https://user-images.githubusercontent.com/628838/197284809-4cc2fa55-f802-402a-945d-82b83502275e.gif" width="368" height="207" />

The principle of Continuity states that an object should not spontaneously appear or disappear. Infants as young as 2.5 months of age are able to reason about Continuity violations. For instance, they detect a violation when an object that is passing behind two occluders positioned a short distance apart fails to become visible between them. Similarly, they detect a violation when a cover is lowered over an object, slid to the side, and then lifted to reveal no object. In our Continuity trials an object may magically appear or disappear under a cover.  
      
### Solidity
<img src="https://user-images.githubusercontent.com/628838/197284963-baa78439-165d-4393-b2e7-ee6ac547986e.gif" width="368" height="207" />

The principle of Solidity states that two solid objects should not be capable of passing through one another. By 3.5 to 4.5 months of age, infants are surprised when a screen rotates through the space occupied by an object in its pat, when a toy car rolls through the space occupied by an object, or when an object is lowered into a container that is then moved forward and to the side to reveal the object standing in its initial position. As shown above, our Solidity trials involve objects which may magically pass through a cover.  
  
### Gravity
<img src="https://user-images.githubusercontent.com/628838/197285079-5e71f6fe-d315-479c-823f-69cd66a921e5.gif" width="368" height="207" />
  
This principle states that unsupported objects should fall. By at least 4.5 months of age, infants are surprised when objects float unsupported. In our physically implausible trials, an object appears to float in the middle of a tube or falls through a closed container.  
  
  
## 🧱 Design principles  
  
InfLevel has several unique design considerations that break convention with usual benchmarks in computer-vision.  
  
### 1. Evaluation only  
  
InfLevel is designed to be _used only for evaluation_. As discussed in our paper, allowing training (even of a linear probe) may result in evaluation metrics confounding physical understanding with feature extraction and matching. This result underlines the subtlety of evaluating the expectations embedded in model representations. We stress that InfLevel is designed to assess the physical reasoning abilities of already trained models;   
we do not allow for any training or fine-tuning on our data as this would lead us to evaluating "what a model can learn" and not "what a model  currently understands." Disallowing training makes InfLevel challenging; this is by design and is how infants are evaluated.  
  
### 2. Easy for humans  
As InfLevel builds on infant experiments, adults evaluated on it obtain near 100% accuracy in our exploratory experiments. Given this  "triviality," a model's failure emphasizes the substantial gap between the physical-reasoning capabilities of infants and artificial agents.  
  
### 3. Controlled  
  
The events depicted in InfLevel videos are highly controlled and mirror events shown to infants. This tight control opens the door for designing heuristics that can obtain high performance. We emphasize that InfLevel is intended to be a diagnostic methodology for studying the physical knowledge embedded in learned representations: a poor performance suggests a fundamental failure of the representation to embed physical principles; high  performance, on the other hand, must be understood in the context of how the model was trained and the inductive biases it employs.  
  
## 📜 Citation  
  
If you use this work, please cite:  
  
```bibtex  
@article{WeihsEtAl2022InfLevel,
  title={Benchmarking Progress to Infant-Level Physical Reasoning in AI},
  author={Luca Weihs and Amanda Rose Yuile and Ren\'{e}e Baillargeon and Cynthia Fisher and Gary Marcus and Roozbeh Mottaghi and Aniruddha Kembhavi},
  journal={TMLR},
  year={2022}
}
```
