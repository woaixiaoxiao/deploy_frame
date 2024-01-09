> kuiperdatawhale推理框架漫游

# Tensor

Tensor作为深度学习的基石，在推理框架中也是必不可少的

在kuiper深度学习框架中，以**Armadillo**线性代数库的矩阵为基础，再它的基础上封装出来了一个Tensor类，具体来说，Tnesor类只有这两个成员

```c++
private:
    std::vector<uint32_t> raw_shapes_; // 张量数据的实际尺寸大小
    arma::fcube data_;                 // 张量数据
```

在各种函数中，需要考虑到底层的**Armadillo**中的矩阵是列主序的，因此如果我们的矩阵是行主序的，需要进行额外的转置操作

同时，有时候如果要直接用原始的指针去操作矩阵中的数据，也要注意返回的指针是列主序的

最后，因为这里封装的都是fcube，因此都是三维矩阵，在图像处理中，一般只能表示一个样本

# 计算图的定义

kuiper框架支持导入PNNX的模型格式，通过如下操作，就可以导入模型，获得所有的operator（操作符）和operand（操作数）

```c++
int load_result = this->graph_->load(param_path_, bin_path_);
```

但是还需要在graph的基础上封装一个运行时graph类，方便在推理的时候构建并计算，具体来说，运行时graph包含了运行时operator和运行时operand

RuntimeOperator的主要成员为

1. params和attribute
   1. params是操作符的超参数，比如linear层的输入和输出维度
   2. attribute是操作符的权重信息，比如linear层的矩阵和偏置
   3. 在kuiper中，各种各样类型的params都有对应的实现（从一个基类params中继承），从int，float到矩阵。而对于存储权重的attribute，只有float32一种存储方法。因此这里留有量化优化的空间
2. 输入操作数和输出操作数（在kuiper中，一个op只会有一个输出，但可以有多个输入，因此它也将一个operand的name设置为生成它的那个op的name）
3. layer层：指向的是在kuiper框架中用C++完成的这个op对应的算子
4. name和type

## 构建运行时graph

通过`bool RuntimeGraph::Init()`完成，在这个函数中，首先将PNNX模型导入普通的graph

1. graph中的所有operator
2. 为operator创建RuntimeOperator
3. 初始化这个RuntimeOperator
   1. input：初始化这个op的所有输入操作数
   2. output：记录这个op输出的操作数会成为哪些op的输入（即建立op之间的先后关系，方便后面进行拓扑排序）
   3. params：各种各样的params，根据情况初始化
   4. attribute：只支持float32类型
4. 最后将这个RuntimeOperator加入到RuntimeGraph的operators_中

# 构建计算图关系和执行顺序

这部分的操作在`void RuntimeGraph::Build(const std::string &input_name, const std::string &output_name) `完成

1. 首先将每个op的后继op给记录到这个op中，比如conv操作后面跟了一个sigmoid操作，那么就要在conv这个op中记录sigmoid是它的后继算子
2. 初始化输入输出空间，其实这部分只需要初始化输出空间，因为当前节点的输入空间，就是它的前继节点的输出
3. 最后根据第1步的顺序关系构建拓扑序列

其实至此，通过Init和Build函数，RuntimeGraph已经构建好了，只要把需要使用的算子补充进来，那就可以完成推理

# KuiperInfer中的算子和注册工厂

在RuntimeGraph中的每个op都有一个layer成员，这个layer成员就是算子。想要新增算子，就要继承Layer类，然后重写这个forward方法

```c++
virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
      std::vector<std::shared_ptr<Tensor<float>>> &outputs);
```

具体来说，Layer和它的Op通过指针相互管理，在Layer中有一个无参数的Foward，在这个函数中，它通过Op准备输入和输出，然后将它们作为参数传入带参数的Forward函数中

## 算子

对于算子layer来说，除了要重写Forward方法实现这个算子的功能之外，还需要提供一个函数来帮助创建这个算子的layer。在kuiper中，这个函数为GetInstance。

以sigmoid算子为例，GetInstance函数作用是创建layer，初始化layer中的参数（sigmoid没有），然后将新创建的layer赋给op中的layer

```c++
ParseParameterAttrStatus SigmoidLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                   std::shared_ptr<Layer> &sigmoid_layer) {
    CHECK(op != nullptr) << "Relu operator is nullptr";
    sigmoid_layer = std::make_shared<SigmoidLayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}
```

## 算子工厂

算子工厂使用了单例模式和工厂模式，其中

`static CreateRegistry &Registry();`返回了工厂，即算子名->算子实现的映射

`static void RegisterCreator(const std::string &layer_type, const Creator &creator);`在工厂中新建了一个算子类型和对应的算子实现

`static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);`就会根据op的layer类型去初始化op中的layer

最后实现了一个工具类`LayerRegistererWrapper`，这个类的构造函数中调用了RegisterCreator，因此，可以在每个算子实现的cpp文件中，初始化一个这个类的对应，就会自动将算子注册到算子工厂中

需要注意的是，算子的名字最好和PNNX中的名字对应

# 各种算子的实现

这个就需要什么就实现什么，比较难的有通过im2col实现的矩阵乘法，以及yolo head。不过总的来说就是一个体力活，也是重难点

# 表达式层

表达式层比较特殊，传入的是一个表达式，需要将它分解成各种操作

1. Tokenizer，将表达式划分成不同的部分
2. Generate_，在之前token的帮助下，构建op和操作数的二叉树
3. ReversePolish，将二叉树转为逆波兰表达式
4. Forward，用栈对逆波兰表达式求值

# 执行模型

在`RuntimeGraph::Forward`中

1. 运行当前op的forward，然后将结果写到后继op的输入中（在Build阶段，只是将op的输入的vector的形状设置了一下，而vector中的元素是智能指针，因此根本没占啥空间。一直等到Forward阶段，前一个op的输出得到了，才会设置当前op的输入指向前一个op的输入）

# 总结

推理框架整体上有三部分

1. init，从PNNX中将各个op读入
2. build，拓扑排序，分配空间，创建算子layer
3. forward，按顺序一个op一个op地执行

这次总结主要是把整体的框架梳理一下，其中一些算子的实现细节，以及Modern C++的精彩用法都没有复习，下次有机会还要深入学习一下各种细节
