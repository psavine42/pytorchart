# PyTorch FlexiLogger

**NOTE: VERY WORK IN PROGRESS**

A utility for visdom metering and logging built on top of TNT, and some utlities for recording model status during training.
The idea is to provide a declarative way to generate meters and loggers with a clean api. 
But isn'y pytorch all about nondeclarative style? 
Well I find the imperative pythonic expirience useful for stuff where I need to expirement, not for the stuff that I know and just need to make go. 
Anyway, the meters and loggers are contained in a single object, which is updated by dictionaries of values.

### Considerations 

If you are not a fan of dictionaries for your data during expirements, this may not be for you.
I tend to like not redoing my logging creation functions for every expriment, so I decided to consolidate these.
If you like tnt's MeterLogger, but also like logging random things like histograms of your weights, this may be for you.


## Examples

### Logger
Api for preset configurations (somewhat inspired by Cadene's pretrainedmodels). 
Adding and logging data uses same api as TNT visdom logger (add, log, reset)

    import flexlogger
    
    # create a logger object
    Stat = flexlogger.get_preset_logger('loss+MSE')
    
    # print out the plots and meters providing their data
    Stat.get_definitions()
    >>> {'loss': ['train_loss', 'test_loss'],
         'mse': ['train_mse', 'test_mse']}
    
    # log stuff one by one
    Stat.add({'train_mse': [torch.randn([4, 20]), torch.randn([4, 20])]})
    Stat.add({'test_mse': [torch.randn([4, 20]), torch.randn([4, 20])]})
    
    # or log several keys at once 
    Stat.add({'train_loss': 0.7, 'test_loss': 0.7]})
    
    # log only keys that you want, and do not reset the meters. 
    Stat.log(epoch, keys=['train_loss', 'train_mse'], reset=False)
    
    # or log all the keys with the None option, and reset the meters by default.
    Stat.log(epoch)
    

Defining custom configured loggers is done with dictionaries, and visdom kwargs. 
These are a bit clunky, but I tend to put my definitions in a file somewhere and reuse all over the place...


    Stat = flexlogger.FlexLogger(
                {'loss': {'type': 'line'}
                 'images': {'type': 'image', opts': {'env': 'my_env2'}}}
                {'train_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
                 'test_loss':  {'type': 'AverageValueMeter', 'target': 'loss'} }
                  env='my_env', uid='my_uid')
                                  
### Model Logging

The TooledModel module registers hooks for recording gradients and model weights during training
When pytorch 1 comes out, will need to add profiling to it maybe. 

    
    inputs, targets = Variable(torch.rand(2, 20)), Variable(torch.rand(2, 3))
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
    
    # Tooling object registers 
    TM = TooledModel(model)
    
    output = model(inputs)
    loss = F.mse_loss(output, targets)
    loss.backward()

    TM.table()
    >>>
    


There are also some utilities that I wrote down one time to remind myself what the plot and meter types are and the shapes of the inputs they take.

    # list all the meters
    flexlogger.meter_types()

    # list all plot types
    flexlogger.plot_types()
    
    # show docs for a meter type
    flexlogger.plot_types()
    
    
    # list all the preset names 
    flexlogger.preset_names()
    
    # show details for a preset config
    flexlogger.preset_info('loss+Acc')
    

## Install 

todo setup.py

## Todo 

1) adding plots on the fly:

    Stat.add_plot()
    Stat.add_meter()
    Stat.update_defintion()

2) maybe a counter for some plots. When each of the values has been filled, plot those values? 

3) Note to self. tooling for a nn.Module for logging weights and debugging

4) is it worth having modes instead of explicit settings for each logger? 
Probably not ... Sine config is declarative as it is, it does not need any utility. 

5) Pass in Klass instead of string? 

6) 