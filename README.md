# 🤖🔜👶❓ Benchmarking Progress to Infant-Level Physical Reasoning in AI  

#### Quick links
* [Website](http://inflevel.allenai.org/)
* [Paper PDF](https://openreview.net/pdf?id=9NjqD9i48M)
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
  
## 💾 Downloading the data

As described in our paper, InfLevel is composed of two parts: InfLevel-Lab (videos filed in an real-world infant cognitive development lab), and InfLevel-Sim (videos generated within the AI2-THOR simulator). You can download these videos from the following links:

1. [InfLevel-Lab Download](https://pub-7320908bcb5b4cdea63c22bc2a38600c.r2.dev/inflevel_lab.tar.gz)
2. [InfLevel-Sim Download](https://pub-7320908bcb5b4cdea63c22bc2a38600c.r2.dev/inflevel_sim.tar.gz)

Once downloaded, you can extract the datasets as usual:
```bash
tar -xvf inflevel_lab.tar.gz
tar -xvf inflevel_sim.tar.gz
```

This will give you the following directory structure:
```
inflevel_lab/
  - continuity/
      center__continuity__darkbluecup__blueclover__ii__LR.mp4
      ...
  - gravity/
      center__gravity__bigbluecup__blackcrayon__ci.mp4
      ...
  - solidity/
      center__solidity__greycup__blueclover__ci.mp4
      ...
inflevel_sim/
  - continuity/
      center__continuity__AlarmClock_1f0ef200__Bowl_Container_7b5a3edb__ii.mp4
      ...
  - gravity/
      ...
  - solidity/
      ...
```

The `inflevel_lab` filenames are structured as follows:
```
<camera_position>__<event_category>__<object1>__<object2>__<event_type>__<direction_of_movement>.mp4
```
Please see the paper for more details on the meaning of the event categories and event types.

The `inflevel_sim` filenames are structured as follows:
```
<camera_position>__<event_category>__<object1>__<object2>__<event_type>.mp4
```

### ⚠️ Filename Warning ⚠️ 

One thing to note is that, in InfLevel-Lab, `<object1>` corresponds to the *primary object* (i.e. the cover) and `<object2>` corresponds to the *secondary object* (i.e. the object that appears or disappears). In InfLevel-Sim, this is reversed.

## 🧪 Evaluating your model

To evaluate your model on InfLevel, please compute scores for each video in the dataset, these scores should be LARGE if your model believes the video is physically plausible and SMALL otherwise. Our evaluation script assumes that you save your for each event category in a `.tsv` (tab-separated values), `.csv`, or `.pkl` (pickle'd pandas dataframe) with the following format:

| camera_loc | cover       | obj        | trial_type | dir | score        |
|------------|-------------|------------|------------|-----|--------------|
| center     | darkbluecup | blueclover | ii         | LR  | -0.842029    |
| center     | darkbluecup | blueclover | ii         | RL  | -0.8803146   |
| center     | darkbluecup | blueclover | iv         | LR  | -0.8198016   |
| center     | darkbluecup | blueclover | iv         | RL  | -0.8407956   |
| center     | darkbluecup | duck       | ii         | LR  | -0.8723224   |
| center     | darkbluecup | duck       | ii         | RL  | -0.8565415   |
| center     | darkbluecup | duck       | iv         | LR  | -0.8439393   |
| center     | darkbluecup | duck       | iv         | RL  | -0.8259625   |
| ...        | ...         | ...        | ...        | ... | ...          |

Note that there will not be a `dir` column when evaluating on InfLevel-Sim.

If, for example, you have saved your results for the continuity subset of InfLevel-Lab at the path
```bash
/path/to/your/continuity/scores.tsv
```
in the above format, you can compute your model's performance by running:
```bash
python evaluator.py \
--scores_df_path /path/to/your/continuity/scores.tsv \
--event_category continuity \
--score_key score # The column in your scores data that contains the model's scores
```
this will print output that may looks like
```python
{'accuracy': 0.4873096446700508, 'pval_1sided': 0.843, 'pval': 0.318}
```
where `accuracy` is the frequency your model scored the videos correctly, `pval_1sided` is the one-sided p-value of `accuracy`, and `pval` is the two-sided p-value of `accuracy`. 

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
