# GOUD.MA: A NEWS ARTICLE DATASET FOR SUMMARIZATION IN MOROCCAN DARIJA

This repo holds the training code for [Goud.ma: a News Dataset for Summarization in Moroccan Darija](https://openreview.net/pdf?id=BMVq5MELb9)

## Dataset
Goud-sum contains 158k articles and their headlines extracted from [Goud.ma](https://www.goud.ma/) news website. The articles are written in the Arabic script. All headlines are in Moroccan Darija, while articles may be in Moroccan Darija, in Modern Standard Arabic, or a mix of both (code-switched Moroccan Darija).

You can find models and dataset on [Goud Hugging Face Organization](https://huggingface.co/Goud).

### Data Splits
| Dataset Split | Number of Instances in Split                |
| ------------- | ------------------------------------------- |
| Train         | 139,288                                     |
| Validation    | 9,497                                       |
| Test          | 9,497                                       |

### Characteristics
|                              | Articles         | Headlines      |
| --------------               | -----------------|--------------  |
| The number of tokens         | 26,780,273       | 2,143,493      | 
| The number of unique tokens  | 1,229,993        | 236,593        |
| Minimum number of tokens     | 32               | 4              |
| Maximum number of tokens     | 6,025            | 74             |
| Average number of tokens     | 169.19           | 13.54          |

## Models
We train encoder-decoder baselines that available on HuggingFace. We warmstart 
the model with available BERT checkpoints and finetune it for the task of Text Summarization.

### Results
The results of warmstarting the encoder and decoder with 3 different BERT checkpoints on the test set.

|BERT checkpoint| ROUGE-1  | ROUGE-2  |  ROUGE-L   |
|---------------|----------|----------|----------- |
|AraBERT        | 23.08    | 8.98      |22.06      |
|DarijaBERT     | 19.41    | 6.64     | 18.48      |
|DziriBERT      | 17.98    | 5.83     | 17.22      |

### Training
The code in this repository can be used to replicate the results presented.

Clone repo
```bash
git clone https://github.com/issam9/goud-summarization-dataset.git
```

Install requirements
```bash
cd goud-summarization-dataset
pip install -r requirements.txt
```

Launch training
```
python train.py
```

config/default.yaml contains config defaults for training the model. You can override these defaults via command line like the following.

```
python train.py trainer.num_epochs=10 generate.num_beams=3
```

### How to use

Models are uploaded to Hugging Face, 

```python
from transformers import EncoderDecoderModel, BertTokenizer

article = """توصل الاتحاد الأوروبي، في وقت مبكر من اليوم السبت، إلى اتفاق تاريخي يستهدف خطاب الكراهية والمعلومات المضللة والمحتويات الضارة الأخرى الموجودة على شبكة الإنترنيت.
وحسب تقارير صحفية، سيجبر القانون شركات التكنولوجيا الكبرى على مراقبة نفسها بشكل أكثر صرامة، ويسهل على المستخدمين الإبلاغ عن المشاكل، ويمكن الاتفاق المنظمين من معاقبة الشركات غير الممتثلة بغرامات تقدر بالملايير.
ويركز الاتفاق على قواعد جديدة تتطلب من شركات التكنولوجيا العملاقة بذل المزيد من الجهد لمراقبة المحتوى على منصاتها ودفع رسوم للجهات المنظمة التي تراقب مدى امتثالها.
ويعد قانون الخدمات الرقمية الشق الثاني من إستراتيجية المفوضة الأوروبية لشؤون المنافسة، مارغريت فيستاغر، للحد من هيمنة وحدة غوغل التابعة لألفابت، وميتا (فيسبوك سابقا) وغيرهما من شركات التكنولوجيا الأمريكية العملاقة.
وقالت فيستاغر في تغريدة “توصلنا إلى اتفاق بشأن قانون الخدمات الرقمية، موضحة أن القانون سيضمن أن ما يعتبر غير قانوني في حالة عدم الاتصال بالشبكة ينظر إليه أيضا ويتم التعامل معه على أنه غير قانوني عبر الشبكة (الإنترنت) – ليس كشعار (ولكن) كواقع”.
وتواجه الشركات بموجب قانون الخدمات الرقمية غرامات تصل إلى 6 في المائة من إجمالي عملياتها على مستوى العالم لانتهاك القواعد بينما قد تؤدي الانتهاكات المتكررة إلى حظرها من ممارسة أعمالها في الاتحاد الأوروبي.
وأيدت دول الاتحاد والمشرعون الشهر الماضي القواعد التي طرحتها فيستاغر والمسماة قانون الأسواق الرقمية التي قد تجبر غوغل وأمازون وأبل وميتا وميكروسوفت على تغيير ممارساتها الأساسية في أوروبا.
"""

tokenizer = BertTokenizer.from_pretrained("Goud/AraBERT-summarization-goud")
model = EncoderDecoderModel.from_pretrained("Goud/AraBERT-summarization-goud")

input_ids = tokenizer(article, return_tensors="pt", truncation=True, padding=True).input_ids
generated = model.generate(input_ids)[0]
output = tokenizer.decode(generated, skip_special_tokens=True) 
```

## Citation

```
@inproceedings{
issam2022goudma,
title={Goud.ma: a News Article Dataset for Summarization in Moroccan Darija},
author={Abderrahmane Issam and Khalil Mrini},
booktitle={3rd Workshop on African Natural Language Processing},
year={2022},
url={https://openreview.net/forum?id=BMVq5MELb9}
}
```

