> 参考教程
>
> - [从零开始一起学习SLAM | 理解图优化，一步步带你看懂g2o代码](https://mp.weixin.qq.com/s/j9h9lT14jCu-VvEPHQhtBw)
> - [从零开始一起学习SLAM | 掌握g2o顶点编程套路](https://mp.weixin.qq.com/s/12V8iloLwVRPahE36OIPcw)
> - [从零开始一起学习SLAM | 掌握g2o边的代码套路](https://mp.weixin.qq.com/s/etFYWaZ6y4XPiXrfqCm53Q)

# 优化器设置

## 步骤

==使用新版本==：老版本不用unique_ptr

- 别名Block

    ```c++
    // p为pose dim，l为landmark dim
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<p,l>> Block; 
    // 也可以动态：Pose和Landmark在程序开始时并不能确定
    // typedef g2o::BlockSolver<g2o::BlockSolverPL<Eigen::Dynamic, Eigen::Dynamic>> BlockX
    ```

- 创建线性求解器

    ```c++
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    ```

> - LinearSolverCholmod ：使用sparse cholesky分解法。继承自LinearSolverCCS
> - LinearSolverCSparse：使用CSparse法。继承自LinearSolverCCS
> - LinearSolverPCG ：使用preconditioned conjugate gradient 法，继承自LinearSolver
> - LinearSolverDense ：使用dense cholesky分解法。继承自LinearSolver
> - LinearSolverEigen： 依赖项只有eigen，使用eigen中sparse Cholesky 求解，因此编译好后可以方便的在其他地方使用，性能和CSparse差不多。继承自LinearSolver

- 创建BlockSolver

    ```c++
    Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));
    ```

- 创建总求解器：从GN、LM、DogLeg 中选一个，这里以LM为例

    ```c++
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    ```

    > g2o::OptimizationAlgorithmGaussNewton
    > g2o::OptimizationAlgorithmLevenberg 
    > g2o::OptimizationAlgorithmDogleg 

- 创建稀疏优化器

    ```c++
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    ```

- 添加点 & 边

- 优化

> 某个时刻只能有一个`unique_ptr`指向其管理的动态内存上的对象。当这个`unique_ptr`销毁时，它所指向的对象也会被销毁。

# 顶点

## 预定义顶点

```c++
VertexSE2 : public BaseVertex<3, SE2>  //2D pose Vertex, (x,y,theta)
VertexSE3 : public BaseVertex<6, Isometry3>  //6d vector (x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion)
VertexPointXY : public BaseVertex<2, Vector2>
VertexPointXYZ : public BaseVertex<3, Vector3>
VertexSBAPointXYZ : public BaseVertex<3, Vector3>

// SE3 Vertex parameterized internally with a transformation matrix and externally with its exponential map
VertexSE3Expmap : public BaseVertex<6, SE3Quat>

// SBACam Vertex, (x,y,z,qw,qx,qy,qz),(x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
// qw is assumed to be positive, otherwise there is an ambiguity in qx,qy,qz as a rotation
VertexCam : public BaseVertex<6, SBACam>

// Sim3 Vertex, (x,y,z,qw,qx,qy,qz),7d vector,(x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
VertexSim3Expmap : public BaseVertex<7, Sim3>
```

## 自定义顶点

```c++
template <int D, typename T>
class BaseVertex: public OptimizableGraph::Vertex{
 public:
  typedef T EstimateType;
  static const int Dimension = D;  ///< dimension of the estimate (minimal) in the manifold space
 protected:
  EstimateType _estimate;
	// ...
}
```

- 设置参数：

    - `D`：vertex的最小维度，
        - 例如，3D空间中旋转是3维的，D = 3
    - `T`：待估计vertex的数据类型，
        - 例如，四元数表达三维旋转，T就是Quaternion 类型

- 重载函数

    ```c++
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    virtual void oplusImpl(const number_t* update);
    virtual void setToOriginImpl();
    ```

    > `read`，`write`：分别是读盘、存盘函数。一般声明一下就好
    >
    > `setToOriginImpl`：顶点重置函数，设定被优化变量的原始值。
    >
    > `oplusImpl`：顶点更新函数，计算优化过程中增量△x 。

- 示例：曲线拟合$y=e^{ax^2+bx+c}$

    ```c++
    class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual void setToOriginImpl(){ // 重置
            _estimate << 0,0,0;
        }
    
        virtual void oplusImpl( const double* update ){ // 更新
            _estimate += Eigen::Vector3d(update);
        }
        // 存盘和读盘：留空
        virtual bool read( istream& in ) {}
        virtual bool write( ostream& out ) const {}
    };
    ```

- 示例：（官方）李代数表示位姿

    ```c++
    /**
     * \brief SE3 Vertex parameterized internally with a transformation matrix
     and externally with its exponential map
     */
    class G2O_TYPES_SBA_API VertexSE3Expmap : public BaseVertex<6, SE3Quat>{
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
      VertexSE3Expmap();
    
      bool read(std::istream& is);
    
      bool write(std::ostream& os) const;
    
      virtual void setToOriginImpl() {
        _estimate = SE3Quat();
      }
    
      virtual void oplusImpl(const number_t* update_)  {	// using number_t = double;
        Eigen::Map<const Vector6> update(update_);
        setEstimate(SE3Quat::exp(update)*estimate());
      }
    };
    ```

## 添加顶点

- 步骤：new一个顶点，添加初始值，设置ID，添加顶点

- 示例：曲线拟合

    ```c++
    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate( Eigen::Vector3d(0,0,0) );
    v->setId(0);
    optimizer.addVertex( v );
    ```

# 边

## 预定义边

## 自定义边

```c++
template <int D, typename E>
class BaseEdge : public OptimizableGraph::Edge{
 protected:
  VertexContainer _vertices;
	// ...
}

template <int D, typename E, typename VertexXi>
class BaseUnaryEdge : public BaseEdge<D,E>{
    // ...
}

template <int D, typename E, typename VertexXi, typename VertexXj>
class BaseBinaryEdge : public BaseEdge<D, E>{
    // ...
}
```

- 类型

    - BaseUnaryEdge：一元边

    - BaseBinaryEdge：两元边

    - BaseMultiEdge：多元边

- 参数

    - D：测量值维度
    - E：测量值类型
    - VertexXi / VertexXj：顶点

- 重载函数

    ```c++
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    virtual void computeError() override {
        // 取出Vertex。。。
        _error = _measurement - Something;
    }     
    virtual void linearizeOplus() override{
        // 取出Vertex。。。
        _jacobianOplusXi(pos, pos) = something;
        // ...         
    }
    ```

## 添加边

- 步骤：new一个边，添加测量值，设置ID，添加顶点，设置协方差矩阵，添加边

- 示例：PnP

    ```c++
    Edge *e = new Edge(p3d, K);
    e->setMeasurement(p2d);
    e->setId(id);
    e->setVertex(num, vertex);	// 一元边只加一个点（num=0），二元边加两个（num=0,1）
    e->setInformation(Eigen::Matrix2d::Identity());	// 不知道的话设为1？
    optimizer.addEdge(e);
    ```

# 优化

```c++
  optimizer.setVerbose(false);			// 关闭调试输出
  optimizer.initializeOptimization();	// 初始化参数
  optimizer.optimize(iterations);		// 设置迭代次数

  cout << v->estimate() << endl;		// 输出结果
```

