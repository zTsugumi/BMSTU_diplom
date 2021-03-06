?	K:???O}@K:???O}@!K:???O}@      ??!       "n
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
??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`??P7?@ym0yE?`X@?"V
8gradient_tape/CapsNet/Encoder/digit_caps/einsum/Einsum_1Einsum?(~?b??!?(~?b??0"w
Kgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?rmw?)??!??,?V??0"F
(CapsNet/Encoder/digit_caps/einsum/EinsumEinsum??uŸ?!_%%??n??0"F
(CapsNet/Encoder/primary_caps/conv/Conv2DConv2D_?~?fh??!??6tE??0"u
Jgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?z??S??!s???????0"N
5gradient_tape/CapsNet/Encoder/digit_caps/einsum/Sum_1Sum?6aw??!+???k???"T
6gradient_tape/CapsNet/Encoder/digit_caps/einsum/EinsumEinsum?a??8??!:?EK0`??0"0
Adam/gradients/AddN_12AddN??/>)???!YY>????"K
2gradient_tape/CapsNet/Encoder/digit_caps/mul_6/SumSum?:??]??!=D?~	??"9
 CapsNet/Encoder/digit_caps/Sum_2Sum?Yb8B??!%????N??Q      Y@Y1@a??????T@qk?}?O@y?????^S?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?63.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 