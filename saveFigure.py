from matplotlib import pyplot as plot

plot.rcParams [ "savefig.facecolor" ] = "w"
plot.rcParams [ "savefig.edgecolor" ] = "b"

lab_value , title_numb = 'Lab_None' , 0

def setLab ( text = None ) :

    global lab_value

    if text : lab_value = text

    print ( 'lab value:' , lab_value )

def save () :

    global title_numb , lab_value

    for i in plot.get_fignums () :

        fig = plot.figure  ( i )

        plot.savefig ( '{}_{}.png'.format ( lab_value , str ( title_numb ) ) )
        print        ( '{}_{}.png'.format ( lab_value , str ( title_numb ) ) )
        
        fig.clear ()
        plot.close ( fig )
        title_numb += 1

def plot_series ( time = [] , series = [] , format = "-" , start = 0   , end  = None ,
                  lr_with_var_value  = [] , history = {} , xmin = None , xmax = None , ymin = None , ymax = None ,
                  title = None , xlabel = None , ylabel = None , labels = [] ) :

    plot.figure ( figsize = ( 10 , 6 ) )
    
    label = ( labels [ 0 ] if ( len ( labels ) and labels [ 0 ] ) else None )

    if len ( time ) and len ( series ) and ( type ( series ) is tuple ) :

      for i , series_num in enumerate ( series ) :

                                           label = ( labels [ i ] if ( len ( labels ) and labels [ i ] ) else None )
                                           plot.plot ( time [ start : end ] , series_num [ start : end ] , format , label = label )
    elif len ( time ) and len ( series ) : plot.plot ( time [ start : end ] , series     [ start : end ] , format , label = label )

    plot.title  ( str ( title ) )
    plot.legend (               )

    plot.xlabel ( "Time"  if not xlabel else xlabel )
    plot.ylabel ( "Value" if not ylabel else ylabel )
    plot.grid   (  True                             )
    
    if len ( lr_with_var_value ) and ( 'loss' in history.history ) and ( 
       xmin != None and xmax != None and ymin != None and ymax != None ) :

       plot.semilogx    ( lr_with_var_value , history.history [ "loss" ]    )
       plot.tick_params ( 'both' , length = 10 , width = 1 , which = 'both' )
       plot.axis        ( [ xmin , xmax , ymin , ymax ]                     )

    save ()
