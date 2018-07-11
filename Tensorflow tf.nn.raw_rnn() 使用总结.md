# tensorflow tf.nn.raw_rnn()使用总结

## api阅读

### raw_rnn()概要

```python
def raw_rnn(cell, loop_fn,
            parallel_iterations=None, swap_memory=False, scope=None):
  """Creates an `RNN` specified by RNNCell `cell` and loop function `loop_fn`.

  **NOTE: This method is still in testing, and the API may change.**

  This function is a more primitive version of `dynamic_rnn` that provides
  more direct access to the inputs each iteration.  It also provides more
  control over when to start and finish reading the sequence, and
  what to emit for the output.

  For example, it can be used to implement the dynamic decoder of a seq2seq
  model.

  Instead of working with `Tensor` objects, most operations work with
  `TensorArray` objects directly.

  The operation of `raw_rnn`, in pseudo-code, is basically the following:

  ```python
  time = tf.constant(0, dtype=tf.int32)
  (finished, next_input, initial_state, emit_structure, loop_state) = loop_fn(
      time=time, cell_output=None, cell_state=None, loop_state=None)
  emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
  state = initial_state
  while not all(finished):
    (output, cell_state) = cell(next_input, state)
    (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
        time=time + 1, cell_output=output, cell_state=cell_state,
        loop_state=loop_state)
    # Emit zeros and copy forward state for minibatch entries that are finished.
    state = tf.where(finished, state, next_state)
    emit = tf.where(finished, tf.zeros_like(emit_structure), emit)
    emit_ta = emit_ta.write(time, emit)
    # If any new minibatch entries are marked as finished, mark these.
    finished = tf.logical_or(finished, next_finished)
    time += 1
  return (emit_ta, state, loop_state)
  ...```

  with the additional properties that output and state may be (possibly nested)
  tuples, as determined by `cell.output_size` and `cell.state_size`, and
  as a result the final `state` and `emit_ta` may themselves be tuples.

  A simple implementation of `dynamic_rnn` via `raw_rnn` looks like this:

  ```python
  inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                          dtype=tf.float32)
  sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
  inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
  inputs_ta = inputs_ta.unstack(inputs)

  cell = tf.contrib.rnn.LSTMCell(num_units)

  def loop_fn(time, cell_output, cell_state, loop_state):
    emit_output = cell_output  # == None for time == 0
    if cell_output is None:  # time == 0
      next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
      next_cell_state = cell_state
    elements_finished = (time >= sequence_length)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        lambda: inputs_ta.read(time))
    next_loop_state = None
    return (elements_finished, next_input, next_cell_state,
            emit_output, next_loop_state)

  outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
  outputs = outputs_ta.stack()
 ... ```
```

- 这个function是非常原始版本的dynamic_rnn,提供了更加直接的到输入迭代的访问接口，也提供了对序列启动 停止读写和进行输出的控制
- Raw_rnn直接操作tensorarray，而不是tensor对象
- docsting中给出了该api使用的简单示例，
- 该函数接受一个非常总要的参数：loop_fn()。该参数控制输入输出。

docstring中的伪代码：
 ```python
  time = tf.constant(0, dtype=tf.int32)
  (finished, next_input, initial_state, emit_structure, loop_state) = loop_fn(
      time=time, cell_output=None, cell_state=None, loop_state=None)
  emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
  state = initial_state
  while not all(finished):
    (output, cell_state) = cell(next_input, state)
    (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
        time=time + 1, cell_output=output, cell_state=cell_state,
        loop_state=loop_state)
    # Emit zeros and copy forward state for minibatch entries that are finished.
    state = tf.where(finished, state, next_state)
    emit = tf.where(finished, tf.zeros_like(emit_structure), emit)
    emit_ta = emit_ta.write(time, emit)
    # If any new minibatch entries are marked as finished, mark these.
    finished = tf.logical_or(finished, next_finished)
    time += 1
  return (emit_ta, state, loop_state)
```

**loop_fn是一个函数，这个函数在rnn的相邻时间步之间被调用。**　　
函数的总体调用过程为：

1. 初始时刻，先调用一次loop_fn，获取第一个时间步的cell的输入，loop_fn中进行读取初始时刻的输入。
2. 进行cell自环　(output, cell_state) = cell(next_input, state)
3. 在t时刻RNN计算结束时，cell有一组输出cell_output和状态cell_state，都是tensor；
4. 到t+1时刻开始进行计算之前，loop_fn被调用，调用的形式为loop_fn( t, cell_output, cell_stat, loop_state)，而被期待的输出为：(finished, next_input, initial_state, emit_output, loop_state)；
5. RNN采用loop_fn返回的next_input作为输入，initial_state作为状态，计算得到新的输出。

在每次执行（output， cell_state） =  cell(next_input, state)后，执行loop_fn()进行数据的准备和处理。

emit_structure 即上文的emit_output将会按照时间存入emit_ta中。 

loop_state  记录rnn loop的变量的状态。用作记录状态 

Tf.where 是用来实现dynamic的。 


### loop_fn()

```python
(elements_finished, next_input, next_cell_state, emit_output, next_loop_state) = loop_fn(time, cell_output, cell_state, loop_state)
```
该函数定义数据转换模式。在调用cell之前，进行转换，准备好输入。 
需要注意的是，该函数的使用要分为两种 情况： 

1. time = 0 时，此时，尚未开始rnn的循环，因此，time=0（time是一个tensorTensor("rnn/Const:0", shape=(), dtype=int32) ）， time, cell_output, cell_state, loop_state =  time, None, None, None 需要进行参数初始化，也就是函数的返回值

    Elements_finished 是一个结束迭代的标志，可以返回（0>sequence_length） 

    Next_input 是输入，可以根据input的tensorArray进行选择传值。也可以自己进行定制化的custom 初始化 

    next_cell_state = cell.zero_state() 或者初始化为想要的值（LSTM的cell需要传入STMstateTuple tensor<c,h>） 

    emit_output 先设置为None  ？？存疑，可以设置projection Layer 。好像并不是，projection Layer 是lstmcell自带的，应该不是这里实现。 作为记录状态使用 

    next_loop_state  None 作为记录状态使用 

2. time = 1~sequence_length时,Elements_finished 是一个结束迭代的标志，可以设置为（time>sequence_length）
    Next_input 是输入，可以根据input的tensorArray进行选择传值。也可以自己进行定制化的custom 得到输入（sequence2sequence 的decoder中使用前一个时刻的输出作为输入） 

    next_cell_state 可以是 cell_state 或者在此基础上定制化 

    emit_output 存疑 用作记录状态

    next_loop_state  用作记录状态

### 使用demo

参考阅读中的第一个

##　参考阅读
- [Why I Use raw_rnn Instead of dynamic_rnn in Tensorflow and So Should You](https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/)
- [知乎问题:有大神能详细讲讲tensorflow中的raw_rnn这个函数么？](https://www.zhihu.com/question/61311860)