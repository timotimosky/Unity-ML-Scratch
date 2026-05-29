using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;


//监督学习的核心有3个函数
//1.拟合函数，本例子就是设定了 float sum = x * weights[0] + y * weights[1] + 1.0f * weights[2];。 这个公式拟合任意线性函数的公式
//2.激活函数，类似如下的一种函数：  f1(sum) = sign(sum) ，将预测值转换为正确和错误标记
//3.权重更新公式（训练算法）
//感知机的权重更新公式是： w = error * x * learningRate

//上面，2和3必须配对，不然会冲突，无法收敛。
//比如如果激活函数是 sum >= 0 → 1，
//那么权重更新公式就必须是 error = target - guess; weights[i] += learningRate * error * input_i;
//这样的形式，才能保证每次调整权重的方向都是正确的。


//在 Unity 场景中随机生成 200 个小球，根据你设定的“秘密直线”为它们涂上正确的颜色，
//然后在 Update 中让感知机不断学习，并通过 LineRenderer 将感知机的“思考过程”实时画出来。
[RequireComponent(typeof(LineRenderer))]
public class TrainingManager : MonoBehaviour
{
    [Header("References")]
    public GameObject pointPrefab;     // 用于显示坐标点的球体预制体
    public LineRenderer lineRenderer;   // 用于画出感知机划分线的 LineRenderer

    [Header("Settings")]
    public int totalPoints = 200;      // 训练点的数量
    public float spawnRange = 5f;      // 随机点的生成范围（-5 到 5）

    // 理想的秘密直线方程: Y = M * X + B
    // 感知机的目标就是通过学习，找出这两个数字！
    [Header("Target Line (Y = M * X + B)")]

    // Y = 0.5X+1;
    // 0= 0.5X-Y+1
    public float m0 = 0.5f;       // 斜率
    public float m1 = 1.0f;       // 截距

    // 内部结构，用于存储每个点的数据
    private class TrainingPoint
    {
        public float x;
        public float y;
        public int targetOutput;       // 正确答案：1（线上方）或 -1（线下方）
        public GameObject visualObj;   // 场景中的小球实例
    }

    private List<TrainingPoint> pointsList = new List<TrainingPoint>();
    protected virtual void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();

        // 配置 LineRenderer 的基础样式
        lineRenderer.positionCount = 2;
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.1f;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = Color.green; // 感知机的线用绿色表示
        lineRenderer.endColor = Color.green;

        // 初始化生成训练数据
        CreateTrainingData();
        InitModel();
    }

    protected virtual void InitModel()
    {        // 实验1: 使用激活函数A (sum >= 0 → 1)
        curPer = new Perceptron();

        // curPer.SetActiveAction(sum => sum >= 0 ? 1 : -1);
        // 实验2: 使用激活函数B (sum <= 0 → 1)
        curPer.SetActiveAction(sum => sum >= 0 ? 1 : -1);
        // 显式在构造外随机初始化权重（因为非Mono类无法自动执行Start）
        for (int i = 0; i < curPer.weights.Length; i++)
        {
            curPer.weights[i] = UnityEngine.Random.Range(-1f, 1f);
        }
    }

    public Perceptron curPer;

    void TestActivationImpactOld()
    {
        // --- 【修改这里】每一帧，我们只训练一个点！ ---
        TrainingPoint pointToTrain = pointsList[currentTrainingIndex];
        curPer.Train(pointToTrain.x, pointToTrain.y, pointToTrain.targetOutput);

        // 更新计数器，指向下一个训练点
        currentTrainingIndex = (currentTrainingIndex + 1) % pointsList.Count;
    }

    public int num = 0;

    // 在类最上方加一个计数器变量
    private int currentTrainingIndex = 0;
    protected virtual void Update()
    {
        num++;
        if (num % 17 == 0)
            ExecuteTraining();

        // 如果感知机猜对了，小球保持原本的红/蓝
        // 如果感知机猜错了，我们可以让它变成黄色，方便观察
        foreach (var point in pointsList)
        {
            int guess = curPer.Guess(point.x, point.y);
            SpriteRenderer renderer = point.visualObj.GetComponent<SpriteRenderer>();

            if (guess != point.targetOutput)
            {
                renderer.color = Color.yellow; // 猜错了，变黄
            }
            else
            {
                renderer.color = (point.targetOutput == 1) ? Color.blue : Color.red; // 猜对了，恢复红蓝
            }
        }

        // 实时画出感知机当前认为的分类线
        DrawPerceptronLine();
        Test();
    }

    void Test()
    {
        // 监听鼠标左键点击
        if (Input.GetMouseButtonDown(0))
        {
            // 获取鼠标在 Unity 2D/3D 空间中的坐标
            Vector3 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mousePos.z = 0; // 确保在 2D 平面上

            // 【核心】此时我们完全不知道正确答案，我们直接问感知机（学生）！
            int prediction = curPer.Guess(mousePos.x, mousePos.y);

            // 动态生成一个新的测试球
            if (pointPrefab != null)
            {
                GameObject newPoint = Instantiate(pointPrefab, mousePos, Quaternion.identity);
                SpriteRenderer renderer = newPoint.GetComponent<SpriteRenderer>(); // 如果是3D Sphere

                // 根据感知机的【自主预测】来上色
                // 如果它预测是 1 就染成蓝色，-1 就染成红色
                if (renderer != null)
                {
                    renderer.color = (prediction == 1) ? Color.blue : Color.red;
                }
            }
        }
    }

    //定义我们要解决的问题的规则
    // 根据秘密直线 Y = M * X + B 计算正确答案
    // 如果点实际的 Y 大于 线的 Y，说明在线上方
    //后续guess函数必须和这个函数保持一致，否则无法收敛
    //提前计算好 真实的Y值
    int ifRight(float X_real, float Y_real)
    {
        float Y_need = m0 * X_real + m1;
        return Y_real > Y_need ? 1 : -1;
    }
    [SerializeField]
    bool OpenChaos = true;

    // 生成随机点并计算正确答案
    void CreateTrainingData()
    {
        for (int i = 0; i < totalPoints; i++)
        {
            TrainingPoint p = new TrainingPoint();
            p.x = Random.Range(-spawnRange, spawnRange);
            p.y = Random.Range(-spawnRange, spawnRange);

            p.targetOutput = ifRight(p.x, p.y);

            if (OpenChaos)
            {
                // --- 核心改动：故意制造 5% 的叛徒（噪声点） ---
                if (Random.value < 0.05f)
                {
                    p.targetOutput = -p.targetOutput; // 强行把线上方的球变成红色，线下的变成蓝色
                }
            }


            // 在场景中生成小球
            if (pointPrefab != null)
            {
                p.visualObj = Instantiate(pointPrefab, new Vector3(p.x, p.y, 0), Quaternion.identity, this.transform);
                // 初始根据正确答案上色
                SpriteRenderer renderer = p.visualObj.GetComponent<SpriteRenderer>();
                if (renderer != null)
                {
                    renderer.color = (p.targetOutput == 1) ? Color.blue : Color.red;
                }
            }

            pointsList.Add(p);
        }
    }

    protected virtual void ExecuteTraining()
    {
        // 原有的轮询单点训练逻辑
        TestActivationImpactOld();
    }


    // 在屏幕上绘制两条线：感知机的线
    void DrawPerceptronLine()
    {
        if (lineRenderer == null) return;

        // 取屏幕左右两端的 X 坐标
        float xLeft = -spawnRange;
        float xRight = spawnRange;

        // 通过感知机当前的权重，计算对应的 Y 坐标
        float yLeft = curPer.GetCurrentLineY(xLeft);
        float yRight = curPer.GetCurrentLineY(xRight);

        // 设置 LineRenderer 的两个端点
        lineRenderer.SetPosition(0, new Vector3(xLeft, yLeft, 0));
        lineRenderer.SetPosition(1, new Vector3(xRight, yRight, 0));
    }

    // 辅助可视化：在 Unity 编辑器的 Scene 窗口直接画出真正的“秘密直线”（红色虚线）
    void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        float xLeft = -spawnRange;
        float xRight = spawnRange;
        float yLeft = m0 * xLeft + m1;
        float yRight = m0 * xRight + m1;
        Gizmos.DrawLine(new Vector3(xLeft, yLeft, 0), new Vector3(xRight, yRight, 0));
    }
}