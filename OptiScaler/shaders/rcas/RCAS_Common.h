#pragma once
#pragma once

#include "SysUtils.h"
#include <d3dcompiler.h>

struct RcasConstants
{
    float Sharpness;

    float MvScaleX;
    float MvScaleY;

    float CameraNear;
    float CameraFar;
};

static std::string daSharpenCode = R"(
#ifdef VK_MODE
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    // Base sharpen amount
    float Sharpness;

    // Depth format
    int DepthIsLinear; // 0 = device/nonlinear, 1 = already linear
    int DepthIsReversed; // 0 = normal Z, 1 = reversed Z

    // Depth rejection
    float DepthScale;
    float DepthBias;

    // Nonlinear depth only.
    // These coefficients must be generated for the non-reversed depth convention.
    // If DepthIsReversed != 0, reversal is applied in shader first:
    // linearDepth = DepthLinearA / max(DepthLinearB - z * DepthLinearC, 1e-6)
    float DepthLinearA;
    float DepthLinearB;
    float DepthLinearC;

    // Motion adaptive
    int DynamicSharpenEnabled;
    int DisplaySizeMV;
    int Debug; // 0=off, 1=motion, 2=depth, 3=combined

    float MotionSharpness; // usually negative
    float MotionTextureScale; // explicit mapping from output pixel space to motion texel space
    float MvScaleX;
    float MvScaleY;
    float MotionThreshold;
    float MotionScaleLimit;

    // Depth mapping from output pixel space to depth texel space
    float DepthTextureScale;

    // Output control
    int ClampOutput;

    // Dimensions
    int DisplayWidth;
    int DisplayHeight;
    int MotionWidth;
    int MotionHeight;
    int DepthWidth;
    int DepthHeight;
};

#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4> Source : register(t0);

#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float2> Motion : register(t1);

#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
Texture2D<float> DepthTex : register(t2);

#ifdef VK_MODE
[[vk::binding(4, 0)]]
#endif
RWTexture2D<float4> Dest : register(u0);

static const int2 kCrossOffsets[4] =
{
    int2(0, -1),
    int2(-1, 0),
    int2(1, 0),
    int2(0, 1)
};

static const int2 kDiagOffsets[4] =
{
    int2(-1, -1),
    int2(1, -1),
    int2(-1, 1),
    int2(1, 1)
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

int2 ClampCoord(int2 p)
{
    return int2(
        clamp(p.x, 0, DisplayWidth - 1),
        clamp(p.y, 0, DisplayHeight - 1)
    );
}

int2 ClampMotionCoord(int2 p)
{
    return int2(
        clamp(p.x, 0, MotionWidth - 1),
        clamp(p.y, 0, MotionHeight - 1)
    );
}

int2 ClampDepthCoord(int2 p)
{
    return int2(
        clamp(p.x, 0, DepthWidth - 1),
        clamp(p.y, 0, DepthHeight - 1)
    );
}

float3 SafeLoadColor(int2 p)
{
    return Source.Load(int3(ClampCoord(p), 0)).rgb;
}

float SafeLoadRawDepthAtCoord(int2 p)
{
    return DepthTex.Load(int3(ClampDepthCoord(p), 0)).r;
}

float2 SafeLoadMotion(int2 p)
{
    return Motion.Load(int3(ClampMotionCoord(p), 0)).rg;
}

float LinearizeDepth(float rawDepth)
{
    float z = rawDepth;

    if (DepthIsLinear > 0)
    {
        if (DepthIsReversed > 0)
            z = 1.0 - z;

        return z;
    }

    if (DepthIsReversed > 0)
    {
        // Non-reversed formula:
        // linear = near * far / (far - z * (far - near))
        //
        // Reversed-Z direct version:
        // zNormal = 1 - zReversed
        // denominator = near + zReversed * (far - near)
        //
        // near = DepthLinearB - DepthLinearC
        float nearPlane = DepthLinearB - DepthLinearC;

        return DepthLinearA / max(nearPlane + z * DepthLinearC, 1e-6);
    }

    return DepthLinearA / max(DepthLinearB - z * DepthLinearC, 1e-6);
}

float SafeLoadDepthLinearFromOutputPixel(int2 pixelCoord)
{
    float2 df = (float2(pixelCoord) + 0.5) * DepthTextureScale;
    int2 depthCoord = int2(df);
    return LinearizeDepth(SafeLoadRawDepthAtCoord(depthCoord));
}

float DistanceSharpnessBoost(float linearDepth)
{
    // Works best if linearDepth is view-space-ish positive distance.
    // log2 keeps the boost gradual and avoids overboosting very far depth.
    float d = max(linearDepth, 1e-4);

    float boost = saturate((log2(d) - 4.0) * 0.15);

    // 1.0 near, up to 1.35 far
    return lerp(1.0, 1.35, boost);
}

float2 EstimateDepthGradient(int2 p, float centerDepth)
{
    float r = SafeLoadDepthLinearFromOutputPixel(p + int2( 1,  0));
    float l = SafeLoadDepthLinearFromOutputPixel(p + int2(-1,  0));
    float u = SafeLoadDepthLinearFromOutputPixel(p + int2( 0,  1));
    float d = SafeLoadDepthLinearFromOutputPixel(p + int2( 0, -1));

    float gxF = r - centerDepth;
    float gxB = centerDepth - l;
    float gyF = u - centerDepth;
    float gyB = centerDepth - d;

    // Prefer the smoother local derivative.
    float gx = abs(gxF) < abs(gxB) ? gxF : gxB;
    float gy = abs(gyF) < abs(gyB) ? gyF : gyB;

    float maxGrad = abs(centerDepth) * 0.05;
    return clamp(float2(gx, gy), -maxGrad, maxGrad);
}

float DepthWeightGrad(float centerDepth, float sampleDepth, float2 gradient, int2 offset)
{
    float predicted = centerDepth + dot(float2(offset), gradient);

    float residual = abs(sampleDepth - predicted);

    // Relative error is much more stable across distance.
    residual /= max(abs(centerDepth), 1e-4);

    residual = max(residual - DepthBias, 0.0);

    return saturate(1.0 - residual * DepthScale);
}

float ComputeAdaptiveSharpness(int2 pixelCoord)
{
    float setSharpness = Sharpness;

    if (DynamicSharpenEnabled > 0)
    {
        float2 mv;

        if (DisplaySizeMV > 0)
        {
            mv = SafeLoadMotion(pixelCoord);
        }
        else
        {
            float2 mvf = (float2(pixelCoord) + 0.5) * MotionTextureScale;
            int2 mvCoord = int2(mvf);
            mv = SafeLoadMotion(mvCoord);
        }

        float motion = max(abs(mv.x * MvScaleX), abs(mv.y * MvScaleY));

        float add = 0.0;

        if (motion > MotionThreshold)
        {
            float denom = max(MotionScaleLimit - MotionThreshold, 1e-6);
            add = ((motion - MotionThreshold) / denom) * MotionSharpness;
        }

        add = clamp(add, min(0.0, MotionSharpness), max(0.0, MotionSharpness));
        setSharpness += add;
    }

    return clamp(setSharpness, 0.0, 2.0);
}

float3 ApplyDebugTint(
    float3 color,
    float baseSharpness,
    float adaptiveSharpness,
    float edgeSharpness,
    float finalSharpness,
    float distanceBoost,
    int debugMode)
{
    float motionBoost = max(adaptiveSharpness - baseSharpness, 0.0);
    float motionReduce = max(baseSharpness - adaptiveSharpness, 0.0);

    // Blue should mean edge-based sharpen reduction only.
    float edgeReduce = max(adaptiveSharpness - edgeSharpness, 0.0);

    float distanceIncrease = max(distanceBoost - 1.0, 0.0);

    if (debugMode > 0)
    {
        color.r *= 1.0 + 12.0 * motionBoost;
        color.r += 0.35 * distanceIncrease;

        color.g *= 1.0 + 12.0 * motionReduce;
        color.b *= 1.0 + 12.0 * edgeReduce;
    }

    return color;
}

float ComputeEdgeFactor(int2 p, float3 center, float centerDepth, float2 depthGrad)
{
    float cLuma   = dot(center, float3(0.2126, 0.7152, 0.0722));
    float lumaSum = 0.0;

    float depthEdge = 1.0;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 o = kCrossOffsets[i];

        float3 tap  = SafeLoadColor(p + o);
        float tLuma = dot(tap, float3(0.2126, 0.7152, 0.0722));
        lumaSum += abs(tLuma - cLuma);

        float tapDepth = SafeLoadDepthLinearFromOutputPixel(p + o);
        float w = DepthWeightGrad(centerDepth, tapDepth, depthGrad, o);

        depthEdge = min(depthEdge, w);
    }

    // Average visible brightness difference around this pixel.
    float lumaAvg = lumaSum * 0.25;

    // 0 = luma does not confirm the depth edge
    // 1 = luma strongly confirms the depth edge
    float lumaConfirm = saturate((lumaAvg - 0.02) * 18.0);

    // Luma is confirmation, not an edge source.
    // Even without luma confirmation, keep some depth protection.
    float depthTrust = lerp(0.15, 1.0, lumaConfirm);

    return lerp(1.0, depthEdge, depthTrust);
}

float ComputeLocalLumaRange(int2 p, float centerLuma)
{
    float lMin = centerLuma;
    float lMax = centerLuma;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float3 tap = SafeLoadColor(p + kCrossOffsets[i]);
        float l = dot(tap, float3(0.2126, 0.7152, 0.0722));
        lMin = min(lMin, l);
        lMax = max(lMax, l);
    }

    return lMax - lMin;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

[numthreads(16, 16, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    int2 p = int2(DTid.xy);

    if (p.x >= DisplayWidth || p.y >= DisplayHeight)
        return;

    float3 c = SafeLoadColor(p);

    float adaptiveSharpness = ComputeAdaptiveSharpness(p);

    if (adaptiveSharpness <= 0.0)
    {
        float3 outColor = c;

        if (Debug > 0)
            outColor = ApplyDebugTint(outColor, Sharpness, adaptiveSharpness, adaptiveSharpness, adaptiveSharpness, 1.0, Debug);

        if (ClampOutput > 0)
            outColor = saturate(outColor);

        Dest[p] = float4(outColor, 1.0);
        return;
    }

    float centerDepth = SafeLoadDepthLinearFromOutputPixel(p);
    float2 depthGrad = EstimateDepthGradient(p, centerDepth);

    // Pre-load cross depths
    float crossDepths[4];
    [unroll]
    for (int i = 0; i < 4; ++i)
        crossDepths[i] = SafeLoadDepthLinearFromOutputPixel(p + kCrossOffsets[i]);

    // Combined luma + depth edge factor
    float edgeFactor = ComputeEdgeFactor(p, c, centerDepth, depthGrad);
    float edgeSharpness = adaptiveSharpness * lerp(0.2, 1.0, edgeFactor);

    float distanceBoost = DistanceSharpnessBoost(centerDepth);
    float motionStability = saturate(adaptiveSharpness / max(Sharpness, 1e-4));
    distanceBoost = lerp(1.0, distanceBoost, motionStability);

    float boostedSharpness = edgeSharpness * distanceBoost;

    float lumaRange = ComputeLocalLumaRange(p, dot(c, float3(0.2126, 0.7152, 0.0722)));

    float unstable = saturate((lumaRange - 0.12) * 4.0);
    unstable *= unstable;

    boostedSharpness *= lerp(1.0, 0.75, unstable);
    float finalSharpness = min(boostedSharpness, 2.0);

    float3 e = c;

    // RCAS 4-neighbor pattern
    float3 b = SafeLoadColor(p + int2( 0, -1));
    float3 d = SafeLoadColor(p + int2(-1,  0));
    float3 f = SafeLoadColor(p + int2( 1,  0));
    float3 h = SafeLoadColor(p + int2( 0,  1));

    // Depth weights for cross taps.
    // crossDepths order matches kCrossOffsets:
    // 0 = up, 1 = left, 2 = right, 3 = down
    float wb = DepthWeightGrad(centerDepth, crossDepths[0], depthGrad, int2( 0, -1));
    float wd = DepthWeightGrad(centerDepth, crossDepths[1], depthGrad, int2(-1,  0));
    float wf = DepthWeightGrad(centerDepth, crossDepths[2], depthGrad, int2( 1,  0));
    float wh = DepthWeightGrad(centerDepth, crossDepths[3], depthGrad, int2( 0,  1));

    // Prevent RCAS from pulling color across depth discontinuities.
    // Unsafe neighbors are blended back toward center.
    b = lerp(e, b, wb);
    d = lerp(e, d, wd);
    f = lerp(e, f, wf);
    h = lerp(e, h, wh);

    // RCAS min/max ring
    float3 minRGB = min(min(b, d), min(f, h));
    float3 maxRGB = max(max(b, d), max(f, h));

    float2 peakC = float2(1.0, -4.0);

    // limiter
    float3 hitMin = minRGB / max(4.0 * maxRGB, 1e-5);
    float3 hitMax = (peakC.xxx - maxRGB) / max(4.0 * minRGB + peakC.yyy, -1e-5);

    float3 lobeRGB = max(-hitMin, hitMax);

    // RCAS is happier with roughly 0..1 range.
    float rcasSharpness = saturate(finalSharpness * 0.75);
    float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * rcasSharpness;
    float rcpL = rcp(4.0 * lobe + 1.0);
    float3 output = ((b + d + f + h) * lobe + e) * rcpL;

    if (Debug > 0)
        output = ApplyDebugTint(output, Sharpness, adaptiveSharpness, edgeSharpness, finalSharpness, distanceBoost, Debug);

    if (ClampOutput > 0)
        output = saturate(output);

    Dest[p] = float4(output, 1.0);
}
)";

static std::string rcasCode = R"(
// Based on this Reshade shader
// https://github.com/RdenBlaauwen/RCAS-for-ReShade

#ifdef VK_MODE
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    float Sharpness;
    float Contrast;

    // Motion Vector Stuff
    int DynamicSharpenEnabled;
    int DisplaySizeMV;
    int Debug;

    float MotionSharpness;
    float MotionTextureScale;
    float MvScaleX;
    float MvScaleY;
    float Threshold;
    float ScaleLimit;
    int DisplayWidth;
    int DisplayHeight;
};

#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float3> Source : register(t0);

#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float2> Motion : register(t1);

#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
RWTexture2D<float3> Dest : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    // Guard against oversized dispatch
    if ((int) DTid.x >= DisplayWidth || (int) DTid.y >= DisplayHeight)
        return;

    int2 pixel = int2(DTid.xy);
    int2 maxPixel = int2(DisplayWidth - 1, DisplayHeight - 1);

    float setSharpness = Sharpness;

    if (DynamicSharpenEnabled > 0)
    {
        float2 mv = float2(0.0, 0.0);
        float motion = 0.0;
        float add = 0.0;

        int2 mvCoord;
        if (DisplaySizeMV > 0)
        {
            mvCoord = pixel;
        }
        else
        {
            mvCoord = int2(pixel * MotionTextureScale);
        }

        // Clamp motion texture reads to valid range
        mvCoord.x = clamp(mvCoord.x, 0, maxPixel.x);
        mvCoord.y = clamp(mvCoord.y, 0, maxPixel.y);

        mv = Motion.Load(int3(mvCoord, 0)).rg;

        motion = max(abs(mv.x * MvScaleX), abs(mv.y * MvScaleY));

        if (motion > Threshold && ScaleLimit > Threshold)
        {
            add = ((motion - Threshold) / (ScaleLimit - Threshold)) * MotionSharpness;
        }

        if ((add > MotionSharpness && MotionSharpness > 0.0) ||
            (add < MotionSharpness && MotionSharpness < 0.0))
        {
            add = MotionSharpness;
        }

        setSharpness += add;
        setSharpness = clamp(setSharpness, 0.0, 1.3);
    }

    float3 e = Source.Load(int3(pixel, 0)).rgb;

    // Skip sharpening if set value == 0
    if (setSharpness == 0.0)
    {
        if (Debug > 0 && DynamicSharpenEnabled > 0 && Sharpness > 0.0)
            e.g *= 1.0 + (12.0 * Sharpness);

        Dest[pixel] = e;
        return;
    }

    // Clamp neighbor accesses at image borders
    int2 coordB = int2(pixel.x, max(pixel.y - 1, 0));
    int2 coordD = int2(max(pixel.x - 1, 0), pixel.y);
    int2 coordF = int2(min(pixel.x + 1, maxPixel.x), pixel.y);
    int2 coordH = int2(pixel.x, min(pixel.y + 1, maxPixel.y));

    float3 b = Source.Load(int3(coordB, 0)).rgb;
    float3 d = Source.Load(int3(coordD, 0)).rgb;
    float3 f = Source.Load(int3(coordF, 0)).rgb;
    float3 h = Source.Load(int3(coordH, 0)).rgb;

    // Min and max of ring
    float3 minRGB = min(min(b, d), min(f, h));
    float3 maxRGB = max(max(b, d), max(f, h));

    float2 peakC = float2(1.0, -4.0);

    // More numerically stable RCAS limiters
    float3 hitMin = minRGB / max(4.0 * maxRGB, 1e-5);
    float3 hitMax = (peakC.xxx - maxRGB) / max(4.0 * minRGB + peakC.yyy, -1e-5);

    float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * setSharpness;

    // Apply contrast adaptation only if Contrast != 0
    if (Contrast != 0.0)
    {
        float3 amp = saturate(min(minRGB, 2.0 - maxRGB) / max(maxRGB, 1e-5));
        amp = rsqrt(max(amp, 1e-5));

        float peak = -3.0 * Contrast + 8.0;
        float contrastFactor = 1.0 / max(amp.g * peak, 1.0);

        lobe *= lerp(1.0, contrastFactor, saturate(Contrast));
    }

    float rcpL = rcp(4.0 * lobe + 1.0);
    float3 output = ((b + d + f + h) * lobe + e) * rcpL;

    if (Debug > 0 && DynamicSharpenEnabled > 0)
    {
        if (Sharpness < setSharpness)
            output.r *= 1.0 + (12.0 * (setSharpness - Sharpness));
        else
            output.g *= 1.0 + (12.0 * (Sharpness - setSharpness));
    }

    Dest[pixel] = output;
}
)";

static ID3DBlob* RCAS_CompileShader(const char* shaderCode, const char* entryPoint, const char* target)
{
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;

    HRESULT hr = D3DCompile(shaderCode, strlen(shaderCode), nullptr, nullptr, nullptr, entryPoint, target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &errorBlob);

    if (FAILED(hr))
    {
        LOG_ERROR("error while compiling shader");

        if (errorBlob)
        {
            LOG_ERROR("error while compiling shader : {0}", (char*) errorBlob->GetBufferPointer());
            errorBlob->Release();
        }

        if (shaderBlob)
            shaderBlob->Release();

        return nullptr;
    }

    if (errorBlob)
        errorBlob->Release();

    return shaderBlob;
}
