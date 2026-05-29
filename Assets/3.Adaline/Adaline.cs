// 继承感知机，重写训练函数，这就是 Adaline 的精髓
public class Adaline : Perceptron
{
    // 重写 Train 函数，将“激活后”的硬误差，改为“激活前”的连续误差
    public override void Train(float x, float y, int target)
    {
        // 1. 【核心差异】：计算激活前的连续净输入净值 sum
        float sum = x * weights[0] + y * weights[1] + 1.0f * weights[2];

        // 2. 【核心差异】：用真实的连续距离去算误差，而不是用 Guess() 激活后的 1 或 -1
        float error = target - sum;

        // 3. 依据均方误差（MSE）对连续误差进行梯度下降更新
        weights[0] += error * x * learningRate;
        weights[1] += error * y * learningRate;
        weights[2] += error * 1.0f * learningRate;
    }
}