	m????~@m????~@!m????~@	iޠֲ?iޠֲ?!iޠֲ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6m????~@Afg?;?@1?+?~@Aam?????Il?f??@Yw?T????*	?E????s@2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2??Kǜg??!??ﬄD@)??Kǜg??1??ﬄD@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?i??_=??!?;?E?B@)?4?($???1?|??6@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?i?q????!'?????1@)?i?q????1'?????1@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ihwH??![?'	t?/@)?ihwH??1[?'	t?/@:Preprocessing2F
Iterator::Model;?O??n??!?!6?g?D@)z??y???1n.??#B
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9jޠֲ?I???????Q?;MКX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Afg?;?@Afg?;?@!Afg?;?@      ??!       "	?+?~@?+?~@!?+?~@*      ??!       2	am?????am?????!am?????:	l?f??@l?f??@!l?f??@B      ??!       J	w?T????w?T????!w?T????R      ??!       Z	w?T????w?T????!w?T????b      ??!       JGPUYjޠֲ?b q???????y?;MКX@