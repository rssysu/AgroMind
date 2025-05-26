# AgroMind

A comprehensive agricultural remote sensing benchmark covering four task dimensions: Spatial Perception, Object Understanding, Scene Understanding, and Scene Reasoning, with a total of 13 task types, ranging from crop identification and health monitoring to environmental analysis. 

![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey)

## 🔗 Link

- **GitHub Pages**: https://rssysu.github.io/AgroMind/
- **Paper(PDF)**: https://arxiv.org/abs/2505.12207
- **Dataset**: https://huggingface.co/datasets/AgroMind/AgroMind
- **Code**: https://github.com/rssysu/AgroMind

## 📂 Structure

```plaintext
AgroMind/
├── AgroMind/
│   ├── models/                 # LLMs    
│   ├── utils/      
│   └── eval.py     
├── QA_Pairs/                   # Tasks
│   ├── Spatial Perception/
│   ├── Object Understanding/
│   ├── Scene Understanding/   
│   └── Scene Reasoning/
├── static/          
│   ├── css/         
│   ├── images/                  # data examples
│   └── js/          
├── .nojekyll    
├── conceptual.pdf               # Project poster
├── README.md                    # introduction
├── index.html                   # GitHub-Page
└── test.txt                     # just for testing
```

## 📌 Key Features
- **Multidimensional Evaluation**
  - 🌍 Spatial Perception
  - 🔍 Object Understanding
  - 🏞️ Scene Understanding
  - 🤖 Scene Reasoning

- **Technical Specifications**
  - 13 specialized agricultural tasks
  - Multimodal data support 

## Dataset
{
  "metadata": {
    "major_task": "Spatial Perception",  // 大任务类型
    "minor_task": "Boundary Detection",   // 子任务类型
    "sensor_type": "satellite"            // 传感器类型
  },
  "items": [
    {
      "qid": "SP-BD-001",                // 问题唯一标识符
      "image_path": "images/satellite/field_001.png",
      "question": "Find the boundaries of cultivated land",
      "answer": {
        "type": "coordinate",
        "value": [0.12, 0.45, 0.87, 0.92]  // [xmin, ymin, xmax, ymax]
      }
    }
  ]
}


## 📜 Cite
```plaintext
@misc{li2025largemultimodalmodelsunderstand,
      title={Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind}, 
      author={Qingmei Li and Yang Zhang and Zurong Mai and Yuhang Chen and Shuohong Lou and Henglian Huang and Jiarui Zhang and Zhiwei Zhang and Yibin Wen and Weijia Li and Haohuan Fu and Jianxi Huang and Juepeng Zheng},
      year={2025},
      eprint={2505.12207},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12207}, 
}
```
