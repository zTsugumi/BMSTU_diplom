	?N??@?N??@!?N??@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?N??@{????J@1??Z@A???=?>??I~?,|@*	7?A`%??@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchge????@!j?v??W@)ge????@1j?v??W@:Preprocessing2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2?n?o?>??!o??O3O@)?n?o?>??1o??O3O@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism	PS???@!?۷g??W@)????·??1(??g?!??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?lscz¢?!???r???)?lscz¢?1???r???:Preprocessing2F
Iterator::Model-$`ty@!?.?#SX@)?#?????1?vS????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?*?b??QU??w?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{????J@{????J@!{????J@      ??!       "	??Z@??Z@!??Z@*      ??!       2	???=?>?????=?>??!???=?>??:	~?,|@~?,|@!~?,|@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?*?b??yU??w?X@