import numpy as np
import theano

THEANO_FLOAT_X = np.dtype(np.double).name
if(hasattr(theano.config,'floatX')):
    THEANO_FLOAT_X = getattr(theano.config,'floatX')

def get_ranges(x):
  mins = x.min(axis = 0)
  maxs = x.max(axis = 0)
  ranges = maxs - mins
  return ranges, mins

def map_to_ranges(x,ranges,mins):
  return ((x - mins) * 2 / ranges) - 1

def map_min_max(x):
  ranges, mins = get_ranges(x)
  return map_to_ranges(x,ranges,mins)

def unmap_y(y_pred,ranges):
  return (y_pred + 1) / (2 * ranges)

def prep_data_np(x,y,split_sizes,randomize):
  do_validation = False
  do_test = True
  train_in = train_out = test_in = test_out = val_in = val_out = None
  if len(x) == 3 and len(y) == 3:
    do_validation = True
    (train_in,val_in, test_in) = x
    (train_out,val_out,test_out) = y
  elif len(x) == 2 and len(y) == 2:
    (train_in, train_out) = x
    (test_in, test_out) = y
  elif type(x) is np.ndarray and type(y) is np.ndarray:
    if(split_sizes is None):
      #assume no split for validation/testing (param not given)
      #just train on the entire given set
      do_test = False
      train_in = x
      train_out = y
    else:
      data_in = x
      data_out = y
      if randomize:
        #randomize whole dataset first
        index = np.random.permutation(np.arange(len(data_in)))
        data_in = data_in[index]
        data_out = data_out[index]
        if((len(split_sizes) != 2 and len(split_sizes) != 3) or
          abs(np.sum(split_sizes) - 1.0) > 2.22e-16 or
          np.min(split_sizes) < 0.0):
            raise ValueError("Expecting an iterable of length 2 or 3 with positive values that add up to 1.0.")
        if(len(split_sizes) == 2):
          split_sizes = (split_sizes[0],0.0,split_sizes[1])
        (train_ratio,val_ratio, test_ratio) = split_sizes
        data_size = len(data_out)
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)
        test_size = data_size - train_size - val_size
        if(val_size > 0):
            do_validation = True
        if(test_size == 0):
            do_test = False
        train_in = data_in[0:train_size]; train_out = data_out[0:train_size]
        val_end = train_size+val_size
        val_in = data_in[train_size:val_end]; val_out = data_out[train_size:val_end]
        test_in = data_in[val_end:]; test_out = data_out[val_end:]
  else:
      raise ValueError("Wrong format for x / y. Expected x and y to be tuples of numpy arrays of lengths 3 or 2, or inidividual numpy arrays.")

  if randomize:
      #randomize train set only (for cases where train set is predefined)
      index = np.arange(len(train_in))
      np.random.shuffle(index)
      train_in = train_in[index]
      train_out = train_out[index]

  return train_in,train_out,val_in,val_out,test_in,test_out,do_validation,do_test

def prep_data(x,y,split_sizes,randomize):
  (x,y,x_v,y_v,x_t,y_t,do_val,do_test) = \
    prep_data_np(x,y,split_sizes,randomize)
  y_ranges, y_mins = get_ranges(y)
  x_ranges, x_mins = get_ranges(x)
  x_c = map_to_ranges(x,x_ranges,x_mins)
  y_c = map_to_ranges(y,y_ranges,y_mins)
  
  if do_val:
    x_v = map_to_ranges(x_v,x_ranges,x_mins)
    y_v = map_to_ranges(y_v,x_ranges,x_mins)
  if do_test:
    x_t = map_to_ranges(x_t,x_ranges,x_mins)
    y_t = map_to_ranges(y_t,x_ranges,x_mins)

  x_c = theano.shared(x_c.astype(THEANO_FLOAT_X),name="x_c",borrow=True)
  y_c = theano.shared(y_c.astype(THEANO_FLOAT_X),name="y_c",borrow=True)

  if(do_val):
      x_v = theano.shared(x_v.astype(THEANO_FLOAT_X),name="x_v",borrow=True)
      y_v = theano.shared(y_v.astype(THEANO_FLOAT_X),name="y_v",borrow=True)
  if(do_test):
      x_t = theano.shared(x_t.astype(THEANO_FLOAT_X),name="x_t",borrow=True)
      y_t = theano.shared(y_t.astype(THEANO_FLOAT_X),name="y_t",borrow=True)

  return x_c,y_c,x_v,y_v,x_t,y_t,do_val,do_test,y_ranges,y_mins,x_ranges,x_mins

