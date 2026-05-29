using UnityEngine;
using System.Collections;

public class TopologyVisualizer : MonoBehaviour
{

    public TrainingManager trainingManager;// 引用你的管理器获取当前训练索引

    [Header("Neuron Visuals")]
    public SpriteRenderer inputXNode;      // 输入层 X 节点
    public SpriteRenderer inputYNode;      // 输入层 Y 节点
    public SpriteRenderer biasNode;        // 偏置项节点
    public SpriteRenderer outputNode;      // 输出层节点

    [Header("Weight Connections")]
    public LineRenderer lineW0;            // 连接 X 和 Output 的线
    public LineRenderer lineW1;            // 连接 Y 和 Output 的线
    public LineRenderer lineW2;            // 连接 Bias 和 Output 的线

    [Header("Visual Settings")]
    [Range(1.5f, 4f)]
    public float thicknessFactor = 1.5f;   // 权重映射到线条粗细的倍数
    public float baseThickness = 0.05f;    // 线条的最小基础粗细
    public float pulseScale = 1.4f;        // 激活时节点的放大倍数

    public float animationSpeed = 8f;      // 节点动画恢复速度

    private Vector3 originalNodeScale = Vector3.one;
    private int lastTrainingIndex = -1;

    void Start()
    {
        trainingManager = FindObjectOfType<TrainingManager>();
        // 自动获取权重连线的两端位置并设置
        SetupLinePositions();

        // 记录节点原始缩放，用于做动画恢复
        if (inputXNode != null) originalNodeScale = inputXNode.transform.localScale;
    }

    void Update()
    {

        // 1. 动态更新连线的粗细与颜色 (读取 weights[0], weights[1], weights[2])
        UpdateWeightLine(lineW0, trainingManager.curPer.weights[0]);
        UpdateWeightLine(lineW1, trainingManager.curPer.weights[1]);
        UpdateWeightLine(lineW2, trainingManager.curPer.weights[2]);

        // 2. 监听训练状态，触发神经元闪烁
        if (trainingManager != null)
        {
            // 通过观察当前训练点的索引是否变化，来判断是否触发了新的前向传播
            int currentIndex = GetPrivateTrainingIndex();
            if (currentIndex != lastTrainingIndex && currentIndex >= 0)
            {
                lastTrainingIndex = currentIndex;
                TriggerPulse();
            }
        }

        // 3. 让节点的缩放和平滑恢复
        SmoothRecoverNodes();
    }

    // 绘制连线
    void SetupLinePositions()
    {
        if (lineW0) { lineW0.positionCount = 2; lineW0.SetPosition(0, inputXNode.transform.position); lineW0.SetPosition(1, outputNode.transform.position); 
            lineW0.material = new Material(Shader.Find("Sprites/Default"));
           }
        if (lineW1) { lineW1.positionCount = 2; lineW1.SetPosition(0, inputYNode.transform.position); lineW1.SetPosition(1, outputNode.transform.position); 
            lineW1.material = new Material(Shader.Find("Sprites/Default"));
           }
        if (lineW2) { lineW2.positionCount = 2; lineW2.SetPosition(0, biasNode.transform.position); lineW2.SetPosition(1, outputNode.transform.position); 
            lineW2.material = new Material(Shader.Find("Sprites/Default"));
           }
    }

    // 根据权重动态改变 LineRenderer
    void UpdateWeightLine(LineRenderer line, float weight)
    {
        if (line == null) return;

        // 方案：基础粗细 + 权重绝对值 * 放大系数
        // 这样即使权重很小，也有根基；权重变大时，粗细拉开明显差距
        float thickness =0.01f+ Mathf.Abs(weight) * thicknessFactor;

        // 稍微放宽上限，允许它变粗到 1.0（在 Size=5 的相机下会非常明显）
        thickness = Mathf.Clamp(thickness, 0.05f, 1.0f);
        Debug.Log("weight====" + weight + " thickness==" + thickness);
        line.startWidth = thickness;
        line.endWidth = thickness;

        // 颜色保持：正数为绿，负数为红
        Color weightColor = weight >= 0 ? Color.green : Color.red;
        line.startColor = weightColor;
        line.endColor = weightColor;
    }

    // 当输入信号流过时，触发节点闪烁
    void TriggerPulse()
    {
        // 节点瞬间放大
        if (inputXNode) inputXNode.transform.localScale = originalNodeScale * pulseScale;
        if (inputYNode) inputYNode.transform.localScale = originalNodeScale * pulseScale;
        if (biasNode) biasNode.transform.localScale = originalNodeScale * pulseScale;

        // 模拟一小段神经传导延迟后，输出节点闪烁
        StopAllCoroutines();
        StartCoroutine(DelayedOutputPulse());
    }

    IEnumerator DelayedOutputPulse()
    {
        yield return new WaitForSeconds(0.05f); // 传导延迟
        if (outputNode) outputNode.transform.localScale = originalNodeScale * pulseScale;
    }

    // 节点尺寸平滑回弹
    void SmoothRecoverNodes()
    {
        Transform[] nodes = { inputXNode?.transform, inputYNode?.transform, biasNode?.transform, outputNode?.transform };
        foreach (var n in nodes)
        {
            if (n != null)
            {
                n.localScale = Vector3.Lerp(n.localScale, originalNodeScale, Time.deltaTime * animationSpeed);
            }
        }
    }

    // 利用反射获取 TrainingManager 里的私有变量 currentTrainingIndex 
    // 这样不需要改动你原本的 TrainingManager.cs 代码
    private int GetPrivateTrainingIndex()
    {
        var field = typeof(TrainingManager).GetField("currentTrainingIndex", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (field != null)
        {
            return (int)field.GetValue(trainingManager);
        }
        return -1;
    }
}