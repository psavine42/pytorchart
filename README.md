# PyTorch FlexiLogger

**NOTE: VERY WORK IN PROGRESS**

A utility for visdom metering and logging built on top of TNT. 
The idea is to provide a declarative way to generate meters and loggers with a clean api.
The meters and loggers are contained in a single object, which is updated by dictionaries of values.

# Considerations 

If you are not a fan of dictionaries for your data during expirements, this may not be for you.
I tend to like not redoing my logging creation functions for every expriment, so I decided to consolidate these.
If you like tnt's MeterLogger, but also like logging random things like histograms of your weights, this may be for you.


# Examples

Api for preset configurations (somewhat inspired by Cadene's pretrainedmodels). 
Adding and logging data uses same api as TNT visdom logger (add, log)

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
    

# Install 

todo

# Todo 

1) adding plots on the fly:

    Stat.add_plot()
    Stat.add_meter()
    Stat.update_defintion()

2) maybe a counter for some plots. When each of the values has been filled, plot those values? 
