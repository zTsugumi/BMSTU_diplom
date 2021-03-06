?	?k???_@?k???_@!?k???_@	???x??????x???!???x???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?k???_@G???Q!@1<?\??[@AQ??ڦx??I?!?uq?@Y7??nf???*	?v??ZX?@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV22?l ]lZ@!N?bc??V@)?l ]lZ@1N?bc??V@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::Shuffle2ޭ,?Yf??!C???9@)ޭ,?Yf??1C???9@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???Xl???!
8?
?@)!????=??11/??P???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?PS???!?r@ş??)?PS???1?r@ş??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch+MJA????!??9
??)+MJA????1??9
??:Preprocessing2F
Iterator::ModeluZ?A????!^ ???@)W?"????1?
????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???x???I??5Y?&@Q??]?j?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	G???Q!@G???Q!@!G???Q!@      ??!       "	<?\??[@<?\??[@!<?\??[@*      ??!       2	Q??ڦx??Q??ڦx??!Q??ڦx??:	?!?uq?@?!?uq?@!?!?uq?@B      ??!       J	7??nf???7??nf???!7??nf???R      ??!       Z	7??nf???7??nf???!7??nf???b      ??!       JGPUY???x???b q??5Y?&@y??]?j?U@?"Y
;gradient_tape/CapsNetMod/Encoder/digit_caps/einsum/Einsum_1Einsum?ImdK???!?ImdK???0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6̪?ڷ?!*e?9????0"o
Dgradient_tape/CapsNetMod/Encoder/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInputJCj????!?z70???0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!D,?e.???0"@
"CapsNetMod/Encoder/conv2d_4/Conv2DConv2DGm<Ts???!?GN:????0"x
Mgradient_tape/CapsNetMod/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?4??$z??!`?Z-????0"o
Dgradient_tape/CapsNetMod/Encoder/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?M/$??!@?=o,??0"@
"CapsNetMod/Encoder/conv2d_3/Conv2DConv2D??8N,??!??R????0"I
+CapsNetMod/Encoder/digit_caps/einsum/EinsumEinsum[U,Ñ??!???n????0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterj?lO???!}??_J??0Q      Y@Y??|??|B@a:?:?O@q???rV@ye???%?t?"?

both?Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 