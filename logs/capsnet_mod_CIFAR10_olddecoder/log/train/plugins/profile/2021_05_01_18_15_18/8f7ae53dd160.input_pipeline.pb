	????=3@????=3@!????=3@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????=3@6<?R6@1}@?3i?*@Ax)u?8F??I???P??@*	????K?m@2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2??7????!a??s?F@)??7????1a??s?F@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchW?oB??!?????3@)W?oB??1?????3@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Lh?XR??!????Z9B@)??ǵ?b??1kQm??0@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?đ"??!$_?Q?&&@)?đ"??1$_?Q?&&@:Preprocessing2F
Iterator::Model?]0?掺?!?sիЮE@)?B ?8???1?*?J??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?11.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIT?.?d?>@Q?]?ŦXQ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6<?R6@6<?R6@!6<?R6@      ??!       "	}@?3i?*@}@?3i?*@!}@?3i?*@*      ??!       2	x)u?8F??x)u?8F??!x)u?8F??:	???P??@???P??@!???P??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qT?.?d?>@y?]?ŦXQ@