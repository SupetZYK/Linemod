# Linemod Note
## Trainer
Hinstere提供的模型都包含颜色信息。使用opengl渲染本身带颜色的模型和不带颜色的模型会有些许差别。
差别主要体现在不用任何light，仅仅使用color_material属性，并使用充足的环境光渲染时，光照条件如下。
```
glLightModelfv(GL_LIGHT_MODEL_AMBIENT,LightAmbient);
glEnable(GL_COLOR_MATERIAL);
glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
```
此时，带颜色的模型会渲染出本身的颜色，而不带颜色的模型渲染的结果是一片纯色。

本工程中使用bu.obj模型和mesh.ply模型渲染的结果不一样。

## 2017-12-12
match 函数中，有一个mask变量似乎无用

```
for (int i = 0; i < (int)modalities.size(); ++i){
    Mat mask, source;
    source = sources[i];
    if(!masks.empty()){
      CV_Assert(masks.size() == modalities.size());
      mask = masks[i];
    }
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modalities[i]->process(source, mask));
  }
```

