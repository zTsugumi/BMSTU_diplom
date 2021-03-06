?	????=3@????=3@!????=3@      ??!       "n
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
	6<?R6@6<?R6@!6<?R6@      ??!       "	}@?3i?*@}@?3i?*@!}@?3i?*@*      ??!       2	x)u?8F??x)u?8F??!x)u?8F??:	???P??@???P??@!???P??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qT?.?d?>@y?]?ŦXQ@?"Y
;gradient_tape/CapsNetMod/Encoder/digit_caps/einsum/Einsum_1EinsumS?ݹ????!S?ݹ????0"x
Mgradient_tape/CapsNetMod/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputo??L???!oM}???0"I
+CapsNetMod/Encoder/digit_caps/einsum/EinsumEinsum?I5f??!?tI????0"o
Dgradient_tape/CapsNetMod/Encoder/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput???????!?P?ٖ???0"o
Dgradient_tape/CapsNetMod/Encoder/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?e?????!{]?X????0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter8$z	??!???`??0"@
"CapsNetMod/Encoder/conv2d_1/Conv2DConv2D??db??!Ӄe????0"@
"CapsNetMod/Encoder/conv2d_2/Conv2DConv2D?w(?$??!kft+?Q??0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?1?â?!U??? ???0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>???!???!???????0Q      Y@Y??18?3@a ??18T@q?Y???V@y'?o??"?
both?Your program is POTENTIALLY input-bound because 19.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?88.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 