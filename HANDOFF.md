# DOTAFS Homepage 续作交接

更新时间：2026-07-25

## 当前成果

首页已经实现一个全屏 2D GPU 流体实验，并提供两个模式：

- `WATER`：WebGPU 2D PBF 水体，支持鼠标击飞、GPU 空间哈希、三轮密度约束和 GPU Metaball 水面。
- `SMOKE`：同一批粒子在水、蒸汽、云和雨之间转换。雨落到水面后会重新成为水粒子，不会创建另一套无关粒子。

天气模式现在采用“共享数据 + 逻辑双池”：

- 所有粒子共享 position、velocity、state 等 GPU Buffer，粒子身份保持不变。
- GPU 每个模拟步把粒子分类到 `WaterActiveList` 或 `CloudActiveList`。
- 两个列表通过 indirect dispatch 只执行各自的活跃粒子数量。
- 水池使用独立空间哈希和完整三轮 PBF。
- 云池使用更大的独立哈希半径和单轮轻量凝聚计算，每个粒子最多采样 64 个邻居。
- 蒸发和降雨只改变状态及索引归属，不复制整批数据，也不进行 CPU 回读。

天气模式实测约有 9,410 个共享粒子。隐藏 Chrome WebGPU 验证中，`GPU SURFACE` 和 `PARTICLES` 两种显示模式都约为 60 FPS。

## 当前已知问题

云彩目前有明显的“向中心抱团”倾向。

原因在 `public/gpu-weather-integrate.wgsl` 的 `cloud_target()`：为了快速形成大块云，当前实现让粒子吸向 3 个固定云锚点。锚点主要分布在画面中间区域，所以云团最终会显得过于集中。

下一步建议：

1. 将 3 个固定锚点改成 6–8 个横向分布的局部云团。
2. 根据粒子 seed 固定所属云团，避免粒子在不同云团之间跳动。
3. 让每个云团拥有独立、缓慢变化的水平漂移相位。
4. 减弱全局锚点吸力，保留局部 cohesion，使云团保持松散而不是收缩成圆球。
5. 同步修改 `SmokeCanvas.tsx` 的初始云粒子分布，让首屏就覆盖更宽区域。

不要重新合并 Water/Cloud 哈希半径，否则云半径变大时会再次拖慢水体 PBF。

## 粒子状态约定

`states[index]` 是一个 `vec4<f32>`：

- `x < -0.5`：水。
- `0.0 .. 1.0`：蒸汽逐渐凝结为云。
- `x > 1.5`：雨。
- `y`：当前生命周期年龄。
- `z`：稳定随机 seed。
- `w`：活跃标记。

## 关键文件

- `app/FluidExperience.tsx`：WATER / SMOKE 模式切换和页面文案。
- `app/FluidCanvas.tsx`：原始 GPU PBF 水模式。
- `app/SmokeCanvas.tsx`：天气模式的资源创建、双池调度、渲染和控制面板。
- `public/gpu-weather-integrate.wgsl`：共享粒子的重力、蒸发、凝结、降雨和鼠标交互。
- `public/gpu-weather-classify.wgsl`：在 GPU 上建立 Water / Cloud 活跃索引及 indirect dispatch 参数。
- `public/gpu-weather-water.wgsl`：水池空间哈希、三轮 PBF、速度重建和黏性。
- `public/gpu-weather-cloud.wgsl`：云池空间哈希、局部凝聚和速度对齐。
- `public/gpu-cloud-density.wgsl`：水、云、蒸汽和雨的 GPU 密度纹理及粒子显示。
- `public/gpu-cloud-surface.wgsl`：卡通风格的 2D 隐式表面。
- `public/gpu-blur.wgsl`：密度纹理的横向、纵向模糊。

`public/gpu-cloud-pbf.wgsl` 是双池改造前的旧统一求解器，目前天气模式不再加载它。可以等新架构稳定后再决定是否删除。

## 回家后运行

要求 Node.js 22.13 或更高版本。

```powershell
cd "Web\DotafsHomepage"
npm install
npm run dev
```

打开终端输出的 Local URL，然后点击页面顶部的 `SMOKE`。

生产构建：

```powershell
npm run build
```

## 验证清单

- 页面没有显示 `GPU CLOUD / WEBGPU UNAVAILABLE`。
- SMOKE 左下状态显示 `GPU DUAL-POOL PBF`。
- 页面只有一个天气 WebGPU canvas。
- `GPU SURFACE` 可以在 `ON` 和 `PARTICLES` 之间切换。
- 鼠标可以击飞水粒子。
- 开启 `WEATHER CYCLE` 后，水会蒸发、凝结、降雨并回到水池。
- 浏览器控制台没有 WGSL 编译或 WebGPU validation error。
- 修改后运行 `npm run build`。

## GitHub 与域名备注

目标 GitHub 仓库：`dotafs2/dotafs2.github.io`。

域名正确拼写是 `dotafsportfolio.com`。之前浏览器错误页中的 `dotafsprotfolio.com` 拼写不一致，多了一个 `t`，会直接触发 `DNS_PROBE_FINISHED_NXDOMAIN`。

如果使用 GitHub Pages 自定义域名，还需要同时确认：

- GitHub Pages 设置中的 Custom domain 是 `dotafsportfolio.com`。
- 仓库发布内容包含正确的 `CNAME`。
- DNS 中 `www` 指向 `dotafs2.github.io`。
- 根域 `@` 的设置与 GitHub Pages 当前要求一致。
