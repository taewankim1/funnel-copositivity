# Funnel using Copositivity LMI Conditions
###  How to Run
To execute the script and see results, use the following command:
```
julia run_result.jl
```
### Output Example
When you run the script, you will see output similar to this:
```
============ Line search for lambda_w ==============
lambda_w: 0.01 cost: diverged
lambda_w: 0.1 cost: diverged
lambda_w: 0.2 cost: diverged
lambda_w: 0.3 cost: -2.2911709525534127
lambda_w: 0.4 cost: -1.9069409141813378
lambda_w: 0.5 cost: -1.4406414895023334
lambda_w: 0.6 cost: -1.0666701375161036
lambda_w: 0.7 cost: -0.7772853920561931
lambda_w: 0.8 cost: -0.5501673803243492
lambda_w: 1.0 cost: -0.18476619077453674
lambda_w: 1.2 cost: 0.34164336496816533
lambda_w: 1.4 cost: diverged
lambda_w: 1.6 cost: diverged
lambda_w: 1.8 cost: diverged
lambda_w: 2.0 cost: diverged
0.3 is picked
======== First copositive condition ========
lambda_w: 0.3 cost: -2.2911709525534127 solve time 0.3278028964996338
======== Second copositive condition ========
lambda_w: 0.3 cost: -2.3700379557202016 solve time 0.7020359039306641
```
### Generated Figure
The script also generates and saves a figure to ./funnels.png, which illustrates the results of the funnels.

The preview of the generated figure is

![Funnel Figure](./funnels.png)


