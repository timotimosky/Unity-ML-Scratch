using System;
using System.Drawing;
using UnityEngine;
/*
 * 感知机去分类 2D 空间中的点 $(x, y)$：目标：随机生成一条直线（比如 $y = 0.5x + 0.1$）。
在线上方的点是“蓝色”，在线下方的是“红色”。感知机的任务：它一开始不知道这条线在哪，
它要通过不断“猜”和“被打脸修正”，最终画出一条一模一样的分割线。
*/
public class Perceptron : MonoBehaviour
{
    // 输入有两个：X 坐标和 Y 坐标，外加一个偏置项（通常固定为 1）
    public float[] weights = new float[3];
    public float learningRate = 0.01f; // 学习率，决定每次犯错后调整的幅度

    public Func<float, int> activeAction = null;

    public void SetActiveAction(Func<float, int> action)
    {
        activeAction = action;
    }

    void Start()
    {
        // 获取主摄像机并设置为正交模式
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.orthographic = true;
            mainCamera.orthographicSize = 7f; // 设置正交大小
        }

        // 1. 初始化权重，先随机给它们一些值
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = UnityEngine.Random.Range(-1f, 1f);
        }
    }

    // 2. 预测函数：输入 X 和 Y，输出 1 或 -1
    public int Guess(float x, float y)
    {
        // 这个公式是 拟合任意线性函数的公式，没有任何问题
        // 加权求和: W0*X + W1*Y + W2*Bias
        float sum = x * weights[0] + y * weights[1] + 1.0f * weights[2];
        return activeAction(sum);
    }

    // 3. 训练函数：根据正确答案调整权重
    public void Train(float x, float y, int target)
    {
        int guess = Guess(x, y);
        int error = target - guess; // 误差 = 正确答案 - 预测值

        // 如果有误差（error != 0），就更新权重
        if (error != 0)
        {
        
            weights[0] += error * x * learningRate;
            weights[1] += error * y * learningRate;
            weights[2] += error * 1.0f * learningRate; // 偏置的权重更新
           // Debug.Log("weights[0]:" + weights[0]);
           // Debug.Log("weights[1]:" + weights[1]);
           // Debug.Log("weights[2]:" + weights[2]);
            //weights[0]:-0.3149419  weights[1]:0.6122891  weights[2]:-0.5972061
        }

        /*-------------------------------------激活函数的设计-------------------------------------

        注意上面的误差更新函数
        因为： weights[1] += error * y * learningRate;
        所以最后的贡献是sum的增量是  sum += y * weights[1] = error * y * y  * learningRate;
        所以我们发现y的正负值不影响sum，只有error的正负值影响sum的增量的正负值。
        所以我们要保持error的

        因为规定了target 是 1（线上方）
        所以error = -2代表 target =-1点在下方，但预测guess=1为在线上方；那么需要调整权重，让guess更倾向为-1。
        而weights[1] += error * y * learningRate; 会变小，会导致sum变小，
         sum = x * weights[0] + y * weights[1] + 1.0f * weights[2];
        所以必须设置激活函数为 sum >= 0 → 1，sum < 0 → -1，
        这样才匹配上了

        -------------------------------------激活函数的设计-------------------------------------*/
    }

    // 辅助函数：根据当前的权重，计算出感知机“眼里”的直线 Y 坐标
    // 权重的方程为: w0*x + w1*y + w2 = 0 -> y = -(w0/w1)*x - (w2/w1)
    public float GetCurrentLineY(float x)
    {
        var weights1 = weights[1];
        if (weights1 == 0) return 0;
        return -(weights[0] * x + weights[2]) / weights1;
    }
}