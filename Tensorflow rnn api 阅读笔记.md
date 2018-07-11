# Tensorflow rnn api 阅读笔记

## 概述

[tensorflow rnn api](https://www.github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn.py)  
rnn api 是用于将rnncell按照时间步骤进行自环的

## 常用rnn

1. 带有static前缀的api要求输入的序列具有固定长度。　　
2. 带有dynamic前缀的api可以选择输入一个sequence_length（可以是一个list）参数,该参数对应的是输入sequence的序列长度，用来动态处理sequence的长度（代码中是设置了一个专门记录序列长度的tensor,控制rnn自环的轮数）。
3. tf.nn.raw_rnn是底层的rnn api，能够使用该api实现各种定制化的操作。很好用。
- tf.nn.static_rnn  
- tf.nn.static_state_saving_rnn  
- tf.nn.static_bidirectional_rnn  
- tf.nn.stack_bidirectional_dynamic_rnn
- tf.nn.dynamic_rnn
- tf.nn.dynamic_bidirectional_rnn
- tf.nn.raw_rnn

### dynamic_rnn

``` python
    def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                    dtype=None, parallel_iterations=None, swap_memory=False,
                    time_major=False, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    Performs fully dynamic unrolling of `inputs`.

    Example:

    ```python
    # create a BasicRNNCell
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

    # defining initial state
    initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

    # 'state' is a tensor of shape [batch_size, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                        initial_state=initial_state,
                                        dtype=tf.float32)
    ```
```
上下两个为官方给的使用样例
```python
  # create 2 LSTMCells
  rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.contrib.rnn.LSTMStateTuple for each cell
  outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
```

```python
  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
      This may also be a (possibly nested) tuple of Tensors satisfying
      this property.  The first two dimensions must match across all the inputs,
      but otherwise the ranks and other shape components may differ.
      In this case, input to `cell` at each time-step will replicate the
      structure of these tuples, except for the time dimension (from which the
      time is taken).
      The input to `cell` at each time step will be a `Tensor` or (possibly
      nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      Used to copy-through state and zero-out outputs when past a batch
      element's sequence length.  So it's more for correctness than performance.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
```

部分参数：　　

```python
parallel_iterations: (Default: 32).  The number of iterations to run in 

      parallel.  Those operations which do not have any temporal dependency 

      and can be run in parallel, will be.  This parameter trades off 

      time for space.  Values >> 1 use more memory but take less time, 

      while smaller values use less memory but computations take longer. 
```

 在迭代过程中控制并行的数量。那些没有时间依赖的并且能够并行的进行的操作　会是这样。该参数能够进行时间和空间的相互转换。 


```python
    swap_memory: Transparently swap the tensors produced in forward inference 

      but needed for back prop from GPU to CPU.  This allows training RNNs 

      which would typically not fit on a single GPU, with very minimal (or no) 

      performance penalty. 
```

显式的交换tensor在前向计算中生成的值，但是需要在cpu中进行反向传播，而不是gpu。这允许了训练那些无法在单个ｇｐｕ上训练的ＲＮＮ，但是会有时间上的延迟。 

前期对inputs数据的transpose工作，(T,B,D) => (B,T,D)　
然后，处理sequence_length转化为对应长度的tensor。
然后，进行rnncell的循环：

```python
    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)
```

该函数的内部执行逻辑为：
首先，确定各个输入参数的状态，  
然后，确定输入的每个time_step的结构是相同的。  
然后，生成各个中间变量的tensorArray，因为是dynamic，所以，使用tensorarray，ta可以看做是具有动态size的tensor数组，通常可以和while_loop或者map_fn结合使用。  
然后，使用tensorflow的控制流control_flow_ops.while_loop函数进行时间步上的循环。

```python
  _, output_final_ta, final_state = control_flow_ops.while_loop(
      cond=lambda time, *_: time < loop_bound,
      body=_time_step,
      loop_vars=(time, output_ta, state),
      parallel_iterations=parallel_iterations,
      maximum_iterations=time_steps,
      swap_memory=swap_memory)

```

可以看到函数的loop的主体是_time_step,该函数已经提前定义，loop_vars是已经定义好的用来做记录的tensorarray。

loop的主体 _time_step()函数如下：
```python
  def _time_step(time, output_ta_t, state):
    """Take a time step of the dynamic RNN.

    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.

    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """
```

中间的执行流程分支分为两个：有sequence_length和无sequence_length的，代码如下：

```python
    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()
```
**核心在于执行rnncell的自环调用过程(output, new_state) = call_cell()**

上面代码块中的_rnn_step()的作用就是根据之前定义的sequence_length的tensor,dynamic的控制rnn的自环的时间步。不再贴代码赘述。

对该过程的讨论也可以参考知乎上以及其他地方的讨论：
[static_rnn和dynamic_rnn的区别](https://www.zhihu.com/question/52200883 )、
[http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)

### bidirectional_dynamic_rnn
```python
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  """Creates a dynamic version of bidirectional recurrent neural network.

  Takes input and builds independent forward and backward RNNs. The input_size
  of forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not
  given.
```

实际上该api的核心在于执行了两次前向和后向dynamic_rnn操作，然后记录各自的cell state 和　hidden state。
如下：

```python
  rnn_cell_impl.assert_like_rnncell("cell_fw", cell_fw)
  rnn_cell_impl.assert_like_rnncell("cell_bw", cell_bw)

  with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

    # Backward direction
    if not time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        return array_ops.reverse(input_, axis=[seq_dim])

    with vs.variable_scope("bw") as bw_scope:
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)
```

### static前缀的rnn

static前缀的rnn的过程类似，而且还会更简单，就不写了

### raw_rnn

raw_rnn是非常灵活且好用的原始rnn api，提供了更加直接的到输入迭代的访问接口，也提供了对序列启动 停止读写和进行输出的控制。使用其他笔记记录。

