# PyTorch FlexiLogger

**NOTE: VERY WORK IN PROGRESS**

A utility for visdom metering and logging built on top of TNT, and some utlities for recording model status during training.
The idea is to provide a declarative way to generate meters and loggers with a clean api. 
But isn'y pytorch all about nondeclarative style? 
Well I find the imperative pythonic expirience useful for stuff where I need to expirement, not for the stuff that I know and just need to make go. 

Or maybe it would be good to just have some common debugging things, and have a way to build plots like [mutual information loss](https://arxiv.org/pdf/1703.00810)
readily available.
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
    
    Stat.add({'train_loss': 0.7, 'test_loss': 0.7]})    # or log several keys at once 
    
    # or log all the keys with the None option, and reset the meters by default.
    Stat.log(epoch)
    Stat.reset()       # clear everythin
    
    
Run a fake training loop ![alt text](imgs/s1.png?raw=true "Title")    

    Stat = flexlogger.get_preset_logger('loss+MSE')
    for i in range(5): # do a loop
        Stat.add({'train_loss': random.random(), 'test_loss': random.random()]})
        Stat.add({k: [torch.randn(4, 20), torch.randn(4, 20)] for k in ['test_mse', 'train_mse']})
        Stat.log(i)
                

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

    # print a table
    TM.table()
    >>> +----------+------------------+------------------+------------------+------------------+
        |          |     Grad_in      |     Grad_out     |      Inputs      |     Weights      |
        | Layers   |   mean     std   |   mean     std   |   mean     std   |   mean     std   |
        +----------+------------------+------------------+------------------+------------------+
        | 0        |  0.0287   0.0068 |  0.0287   0.0068 |  0.2769   0.4568 |  0.1318  -0.0042 |
        | 1        |  0.0519  -0.0658 |  0.0519  -0.0658 |  0.2946  -0.0271 |  0.1878  -0.0139 |
        +----------+------------------+------------------+------------------+------------------+
    
    # or return the dictionary of values for whatever:
    TM.table()

### Why Not Both? 

The TooledModelLogger class creates a ModelLogger, and then send that to a FlexiLogger. 
TooledMOdelLogger has same APIs, (log, reset, add) with the addition of step. 
For now, every time you want to move data, you should call TooledModelLogger.step(). 
I will play around a bit on how this wants to be configured and initialized, but for now TML
will create a chart for each layer or for each metric. 

    from src.modellogger import TooledModelLogger
    
    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
    optim = torch.optim.Adam(model.parameters())

    TMLogger = TooledModelLogger(model)
    for i in range(4):
        inputs = Variable(torch.rand(2, 20))
        targets = Variable(torch.rand(2, 3))
        
        outputs = model(inputs)
        (F.mse_loss(outputs, targets)).backward()
        optim.step()
        
        # call step to iterate its step counter, and send data to loggers.
        # log indicates to flush the Loggers into their plots
        TMLogger.step(log=True)
        
Results in:

![alt text](imgs/s2.png?raw=true "Title")

See examples file for mnist example, and the unittests for options on plot inits. Really I think my goal is to have these config files
for common tasks that work across all my models (like tensorboard, but with control over enviornments and customization )

### Misc Notes
There are also some utilities that I wrote down one time to remind myself what the plot and meter types are and the shapes of the inputs they take.

    
    flexlogger.meter_types()    # list all the meters
    flexlogger.plot_types()     # list all plot types
    flexlogger.plot_types()     # show docs for a meter type
    
    flexlogger.preset_names()           # list all the preset names 
    flexlogger.preset_info('loss+Acc')  # show details for a preset config
    

### Install 
todo setup.py

    pip install visdom
    pip install --upgrade git+https://github.com/psavine42/flexilogger.git@master


### Todo 

1) adding plots on the fly:


    Stat.add_plot()
    Stat.add_meter()
    Stat.update_defintion()

4) is it worth having modes instead of explicit settings for each logger? 
Probably not ... Sine config is declarative as it is, it does not need any utility. 

5) Pass in Klass instead of string? 

5) more tests

6) decide which config layout to use.

7) visdom seriazliation tools for test_check

8) do something 

