# EmpHi

## EmpHi: Generating Empathetic Responses with Human-like Intents

### See Appendix in appendix.pdf

Requirement:
- Python 3.8+
- Parlai 1.0.0
- Torch

para.pkl [Click here to download]([http://www.cnblogs.com/sxdcgaq8080/p/7894828.html](https://drive.google.com/file/d/1OCJaZGGT7vTW4TAjJxkOXglKSqfTR1Gy/view?usp=sharing))

Train Emphi by runing the command below:
```
python train_emphi.py
```

Eval Emphi by runing the command below:
```
python eval.py
```

Hopefully, you will reproduced the following results:

| Methods      | BLEU F1   | Distinct-1     | Distinct-2     |
| ---------- | :-----------:  | :-----------: | :-----------: |
| Multitask-Trans     | 0.3526     | 0.4123     | 1.1390     |
| MoEL     | 0.3298     | 0.8473     | 4.4698     |
| MIME     | 0.3240     | 0.3952     | 1.3299     |
| EmpHi     | 0.3820     | 1.1188     | 5.3332     |
