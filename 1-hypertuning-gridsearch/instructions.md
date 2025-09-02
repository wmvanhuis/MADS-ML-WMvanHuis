# Report
## 1. experiment
Experiment with things like:
- change the number of epochs, eg to 5 or 10.
- changing the amount of units1 and units2 to values between 16 and 1024. Use factors of 2 to easily scan the ranges: 16, 32, 64, etc.
- changing the batchsize to values between 4 and 128. Again, use factors of two for convenience.
- change the depth of your model by adding a additional linear layer + activation function
- changing the learningrate to values between 1e-2 and 1e-5
- changing the optimizer from SGD to one of the other available algoritms at [torch](https://pytorch.org/docs/stable/optim.html) (scroll down for the algorithms)

Check the results:
- all your experiments are saved in the `modellogs` directory, with a timestamp. Inside you find a model.toml file, that
contains all the settings of the model. The `events` file is what tensorboard will show.
- visualize the relationship between variables: for example, make a heatmap of units vs layers.

Studyquestions:
- Epochs: what is the upside, what is the downside of increasing epochs? Do you need more epochs to find out which configuration is best? When do you need that, when not?
- what is an upside of using factors of 2 for hypertuning? What is a downside?

## Note
A note on train_steps: this is a setting that determines how often you get an update.
Because our complete dataset is 938 (60000 / 64) batches long, you will need 938 trainstep to cover the complete 60.000 images.

This can actually be a bit confusion for some students, because changing the value of trainsteps 938 changes the meaning of an `epoch` slightly, because one epoch is no longer the full dataset, but simply `trainstep` batches. Setting trainsteps to 100 means you need to wait twice as long before you get feedback on the performance, as compared to trainsteps=50. You will see that settings trainsteps to 100 improves the learning, but that is simply because the model has seen twice as much examples as compared to trainsteps=50.

This implies that it is not usefull to compare trainsteps=50 and trainsteps=100, because setting it to 100 will always be better.
Just pick an amount that works for your hardware & patience, and adjust your number of epochs accordingly (increase epochs with lower values for trainsteps)

# 2. Reflect
Doing a master means you don't just start engineering a pipeline, but you need to reflect. Why do you see the results you see? What does this mean, considering the theory? Write down lessons learned and reflections, based on experimental results. This is the `science` part of `data science`.

You follow this cycle:
- make a hypothesis
- design an experiment
- run the experiment
- analyze the results and draw conclusions
- repeat

## Tip
To keep track of this process, it is useful to keep a journal. While you could use anything to do so, a nice command line tool is [jrnl](https://jrnl.sh/en/stable/). This gives you the advantage of staying in the terminal, just type down your ideas during the process, and you can always look back at what you have done.
Try to first formulate a hypothesis, and then design an experiment to test it. This will help you to stay focused on the goal, and not get lost in the data.

Important: the report you write is NOT the same as your journal! The journal will help you to keep track of your process, and later write down a reflection on what you have done where you draw conclusion, reflecting back on the theory.

# 3. Make a short report
Make a short report of your findings.
pay attention to:
- what was your hypothesis about interaction between hyperparameters?
- what did you find?
- visualise your results about the relationship between hyperparameters.
