using UnityEngine;

public class XORNet
{
    // 权重矩阵：w_ih (输入到隐藏), w_ho (隐藏到输出)
    public float[,] w_ih = new float[2, 2];
    public float[] w_ho = new float[2];

    // 偏置项
    public float[] b_h = new float[2];
    public float b_o;

    public float learningRate = 0.15f;

    // 暂存激活值，用于反向传播求导
    private float[] h_act = new float[2];
    private float o_act;

    public XORNet()
    {
        // 采用较小的随机初始化，防止 Sigmoid 进入饱和区
        for (int i = 0; i < 2; i++)
        {
            b_h[i] = Random.Range(-0.5f, 0.5f);
            w_ho[i] = Random.Range(-0.5f, 0.5f);
            for (int j = 0; j < 2; j++)
                w_ih[i, j] = Random.Range(-0.5f, 0.5f);
        }
        b_o = Random.Range(-0.5f, 0.5f);
    }

    // 激活函数使用 Sigmoid，确保梯度连续可导
    private float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));
    private float SigmoidDeriv(float val) => val * (1f - val);

    // 前向传播
    public float Forward(float x1, float x2)
    {
        // 输入层 -> 隐藏层
        for (int i = 0; i < 2; i++)
        {
            float sum = x1 * w_ih[0, i] + x2 * w_ih[1, i] + b_h[i];
            h_act[i] = Sigmoid(sum);
        }
        // 隐藏层 -> 输出层
        float finalSum = h_act[0] * w_ho[0] + h_act[1] * w_ho[1] + b_o;
        o_act = Sigmoid(finalSum);
        return o_act;
    }

    // 反向传播（链式法则的核心）
    public void Train(float x1, float x2, float target)
    {
        float guess = Forward(x1, x2);

        // 1. 计算输出层误差梯度
        float error_o = target - guess;
        float delta_o = error_o * SigmoidDeriv(o_act);

        // 2. 计算隐藏层误差梯度（误差回传）
        float[] delta_h = new float[2];
        for (int i = 0; i < 2; i++)
        {
            delta_h[i] = delta_o * w_ho[i] * SigmoidDeriv(h_act[i]);
        }

        // 3. 更新权重与偏置
        for (int i = 0; i < 2; i++)
        {
            w_ho[i] += delta_o * h_act[i] * learningRate;
            w_ih[0, i] += delta_h[i] * x1 * learningRate;
            w_ih[1, i] += delta_h[i] * x2 * learningRate;
            b_h[i] += delta_h[i] * learningRate;
        }
        b_o += delta_o * learningRate;
    }
}