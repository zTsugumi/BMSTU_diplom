	K:???O}@K:???O}@!K:???O}@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-K:???O}@???N?!@1??_Y)?|@A?].?;1??IYiR
??@*	????xAm@2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2@j'???!?_??XVG@)@j'???1?_??XVG@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetche5]Ot]??!o?-p/U4@)e5]Ot]??1o?-p/U4@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?St$????!?.???A@)?q??????1{_?sm-@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??6 ??!??hz,@)??6 ??1??hz,@:Preprocessing2F
Iterator::Model??<?;k??!????C@)?D?$]??1??j?(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI`??P7?@Qm0yE?`X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???N?!@???N?!@!???N?!@      ??!       "	??_Y)?|@??_Y)?|@!??_Y)?|@*      ??!       2	?].?;1???].?;1??!?].?;1??:	YiR
??@YiR
??@!YiR
??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`??P7?@ym0yE?`X@