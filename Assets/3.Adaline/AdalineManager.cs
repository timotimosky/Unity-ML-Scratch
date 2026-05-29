using UnityEngine;

// 派生自 Demo1 的管理器，复用所有的小球生成、LineRenderer 绘制和鼠标点击交互测试逻辑
public class AdalineManager : TrainingManager
{
    [Header("Adaline 特有配置")]
    public float adalineLR = 0.005f; // Adaline 需要更小的学习率以防连续梯度震荡
    public bool enableDecay = true;  // 是否开启学习率衰减

    // 1. 重写初始化，将 Perceptron 掉包为 Adaline
    protected override void InitModel()
    {
        curPer = new Adaline(); // 多态：用子类实例填充基类引用

        // 初始化权重
        for (int i = 0; i < curPer.weights.Length; i++)
        {
            curPer.weights[i] = Random.Range(-0.5f, 0.5f);
        }

        curPer.learningRate = adalineLR;
        curPer.SetActiveAction(sum => sum >= 0 ? 1 : -1); // 激活函数保持完全一致
    }

    // 改进1：学习率衰减（Learning Rate Decay）。
    //前期（冷启动）：学习率大（0.005f），分类线像脱缰的野马，快速在屏幕上奔跑，寻找大致的山谷方位。
    //后期（微调期）：随着帧数推进，步长被逐渐剥离、不断收缩。当线无限接近真正的谷底时，它的针脚变得极度细密，从而彻底消灭了窄幅抖动，优雅地凝固在绝对最优解上。
    //在现代深度学习（如训练大模型或图片分类 MLP）中，这被称为 学习率调度器（LR Scheduler），是让上百亿参数的网络最终安全收敛的工业标准配置。

    /*改进2：从“批量梯度下降（Batch GD）”到“高吞吐的随机梯度下降（SGD）”
   1960年原版：最初的 LMS（Least Mean Squares）算法在理论推导时，倾向于使用批量梯度下降——即把所有准备好的训练样本全部打包喂给数学公式，在纸上算出一个综合的、最完美的总体梯度后，权重才小心翼翼地移动一步。这种做法数学上很严谨，但是在处理海量现实数据（比如几万张图片）时，内存会瞬间爆炸，计算极其缓慢。
   这一版：直接采用了现代分布式计算和游戏引擎最爱的 随机梯度下降（SGD） 机制。利用 Unity 的 Update() 生命周期作为隐式时钟，
    每一帧（每一个 Tick）只抓取一个随机点进行极速修正。
  虽然这种“抓到一个点就打脸修正一次”的方式在前期会带来分类线的曲折试探，但它的计算吞吐量极高，不仅天然具备逃离局部极小值（Local Minima）的潜力，
    更是当今各种神经网络大模型在显卡（GPU）上能够跑起来的核心基石。*/
    protected override void Update()
    {
       // 在原本的 1960 年第一版 Adaline 上，维德罗和霍夫设定的学习率LearningRate是一个完全固定不变的常数。
       // 通过加入衰减系数（curPer.learningRate *= 0.9995f），成功解决了随机梯度下降（SGD）在谷底无法精确收敛、只能在极限环来回震荡的世纪难题。

        // 动态让学习率随时间逐渐缩水，消灭 SGD 的“窄幅抖动”，让线走向绝对静止
        if (enableDecay && curPer != null && curPer.learningRate > 0.00001f)
        {
            curPer.learningRate *= 0.9995f;
        }

        // 调用基类的 Update，完美复用原本的颜色刷新、画线和鼠标点击预测逻辑
        base.Update();
    }
}