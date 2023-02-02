Scientifically there's a correlation between less amount of sleep and tending to overeat. As I'm on a diet right now, I have data from these sources:

- Autosleep (Sleep tracking app)
- Macrofactor (Dieting app)

My objective is to train a ML algorithm that is capable of predicting if I will overeat or keep my calories as close as possible to the objective based on the following data:
- Hours slept
- Caloric intake for the day
- Calorie objective

The model used is a logistic regression model, which is capable of predicting the probability of overeating happening giving them an amount of hours slept and the target calories for the day. It can also calculate the probability and tell you how confident it is about you overeating or not.
