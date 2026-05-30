using UnityEngine;
using System.Collections.Generic;

public class XORVisualizer : MonoBehaviour
{
    [Header("References")]
    public GameObject pointPrefab; // 用于显示坐标点的球体预制体

    [Header("Visualization Settings")]
    public float pointSpacing = 4f; // 空间放大系数：将逻辑上的 1 个单位放大到 4 米
    public float centerOffset = 2f;  // 平移偏移量：让四个点围绕屏幕 (0,0) 中心对称

    [Header("Grid Visualization (Gizmos)")]
    [Range(10, 50)] public int gridResolution = 30; // 网格分辨率，越大背景热力图越细腻
    public float gizmoSampleSize = 6f;             // 采样区域半径大小

    // 内部点数据结构，彻底解耦“数学真理”与“渲染位置”
    private class XORPoint
    {
        public float logicalX; // 逻辑输入：0 或 1（喂给神经网络）
        public float logicalY; // 逻辑输入：0 或 1（喂给神经网络）
        public float target;   // 标签答案：0 或 1
        public GameObject visualObj; // 场景中的物理球体
    }

    private List<XORPoint> xorPointsList = new List<XORPoint>();
    private XORNet xorNetwork;

    void Start()
    {
        // 1. 强制将主摄像机调整为适合观察的尺寸
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            mainCamera.orthographic = true;
            mainCamera.orthographicSize = 5f;
        }

        // 2. 初始化多层感知机网络
        xorNetwork = new XORNet();

        // 3. 构造标准的 XOR（异或）四个核心真理点
        CreateXORPoint(0f, 0f, 0f); // 左下：红
        CreateXORPoint(1f, 1f, 0f); // 右上：红
        CreateXORPoint(0f, 1f, 1f); // 左上：蓝
        CreateXORPoint(1f, 0f, 1f); // 右下：蓝
    }

    void Update()
    {
        if (xorNetwork == null || xorPointsList.Count == 0) return;

        // 1. 【高频训练阶段】：每一帧随机抽取样本让网络在后台内卷 150 次
        for (int i = 0; i < 150; i++)
        {
            int randomIndex = Random.Range(0, xorPointsList.Count);
            XORPoint trainTarget = xorPointsList[randomIndex];

            // 喂给训练器的永远是逻辑上的 0 和 1
            xorNetwork.Train(trainTarget.logicalX, trainTarget.logicalY, trainTarget.target);
        }

        // 2. 【渲染刷新阶段】：利用网络当前的预测值，实时动态过渡小球的颜色
        foreach (var point in xorPointsList)
        {
            float guessOutput = xorNetwork.Forward(point.logicalX, point.logicalY);

            SpriteRenderer renderer = point.visualObj.GetComponent<SpriteRenderer>();
            if (renderer != null)
            {
                // 利用 Color.Lerp 实现从纯红(0) 到 纯蓝(1) 的平滑色彩过渡
                renderer.color = Color.Lerp(Color.red, Color.blue, guessOutput);
            }
        }

        // 3. 【实时交互阶段】：监听玩家点击屏幕进行动态预测
        TestMouseClickPrediction();
    }

    /// <summary>
    /// 监听鼠标点击，在任意位置生成测试球并让网络实时预测预测
    /// </summary>
    private void TestMouseClickPrediction()
    {
        if (Input.GetMouseButtonDown(0)) // 监听鼠标左键
        {
            if (Camera.main == null || pointPrefab == null) return;

            // 获取鼠标在 Unity 世界空间中的点击位置
            Vector3 mouseWorldPos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            mouseWorldPos.z = 0f; // 锁定在 2D 平面

            // ⚠️【核心硬核改动】：逆向映射！
            // 必须把点击的世界坐标 (例如 2.4, -1.2) 还原成网络认识的逻辑坐标 [0, 1] 区间
            float logX = (mouseWorldPos.x + centerOffset) / pointSpacing;
            float logY = (mouseWorldPos.y + centerOffset) / pointSpacing;

            // 让网络基于这组逻辑坐标，自主在前向传播中推算结果概率（0.0 ~ 1.0）
            float prediction = xorNetwork.Forward(logX, logY);

            // 在鼠标点击的物理位置实例化一个新的测试球
            GameObject newTestPoint = Instantiate(pointPrefab, mouseWorldPos, Quaternion.identity);
            newTestPoint.name = $"User_Test_({logX:F2},{logY:F2})";

            SpriteRenderer renderer = newTestPoint.GetComponent<SpriteRenderer>();
            if (renderer != null)
            {
                // 根据多层网络的智能预测进行连续色彩过渡染色
                renderer.color = Color.Lerp(Color.red, Color.blue, prediction);
            }
        }
    }

    /// <summary>
    /// 创建并映射 XOR 核心真理点的辅助函数
    /// </summary>
    private void CreateXORPoint(float logX, float logY, float targetLabel)
    {
        XORPoint newPoint = new XORPoint();
        newPoint.logicalX = logX;
        newPoint.logicalY = logY;
        newPoint.target = targetLabel;

        // 正向映射：将逻辑坐标 [0, 1] 放大平移到 Unity 世界观测区
        float worldX = logX * pointSpacing - centerOffset;
        float worldY = logY * pointSpacing - centerOffset;
        Vector3 worldPos = new Vector3(worldX, worldY, 0f);

        if (pointPrefab != null)
        {
            newPoint.visualObj = Instantiate(pointPrefab, worldPos, Quaternion.identity, this.transform);
            newPoint.visualObj.name = $"XOR_True_Point_({logX},{logY})";

            SpriteRenderer renderer = newPoint.visualObj.GetComponent<SpriteRenderer>();
            if (renderer != null) renderer.color = Color.gray; // 开局混沌灰色
        }

        xorPointsList.Add(newPoint);
    }

    /// <summary>
    /// 利用 Gizmos 在 Scene 窗口实时把多层网络的非线性“思维热力图”画出来
    /// </summary>
    void OnDrawGizmos()
    {
        if (xorNetwork == null) return;

        float step = (gizmoSampleSize * 2) / gridResolution;

        for (int i = 0; i < gridResolution; i++)
        {
            for (int j = 0; j < gridResolution; j++)
            {
                float worldX = -gizmoSampleSize + i * step + step * 0.5f;
                float worldY = -gizmoSampleSize + j * step + step * 0.5f;

                // 逆向映射：世界物理坐标 -> 逻辑网络输入
                float logX = (worldX + centerOffset) / pointSpacing;
                float logY = (worldY + centerOffset) / pointSpacing;

                float prediction = xorNetwork.Forward(logX, logY);

                Color fieldColor = Color.Lerp(Color.red, Color.blue, prediction);
                fieldColor.a = 0.2f; // 20% 透明度背景
                Gizmos.color = fieldColor;

                Gizmos.DrawCube(new Vector3(worldX, worldY, -0.1f), new Vector3(step * 0.9f, step * 0.9f, 0.01f));
            }
        }
    }
}