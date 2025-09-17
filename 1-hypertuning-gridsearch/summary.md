# Summary week 1

First run the notebook as is and studied the results in Tensorboard. A lot of lines in the linegraphs. I find it difficult to understand what I am looking at, can you please explain?

After that a run the notebook again and added the changes as mentioned in the notebook by adding a units3 class.
The results is that in the model.toml an extra attribute is added with datatype int.
I now get an error at the trainer. 

Second try with adding units3 it worked correctly. I now see extra output for units3.

A third experiment I changed the epoch to 5. This makes the model a bit slower, because it trains the model more than it was before (epochs=3).
Looking at the loss/test it looks similar like before, looking at the loss/train it is now a bit lower than before. Accuracy seems to be the same as before.

A fourth experiment I changed the batchsize to 128 (from 64), changed the units1/2/3 to 512 (from 256).
Changed units = [512, 256, 128] (from units = [256, 128, 64]).
Added units3 to the trainer loop. Forget this in previous steps.
This process now runs much longer.
The loss/test is now lower, which is good.
The loss/train seems the same as before.
The accuracy is better, now around 0.88, before 0.84.

Result
With adding more epochs and units the errors became lower and the accuracy became higher.


Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
