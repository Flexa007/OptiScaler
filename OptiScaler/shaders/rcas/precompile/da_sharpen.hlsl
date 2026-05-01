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
    return int2(clamp(p.x, 0, DisplayWidth - 1), clamp(p.y, 0, DisplayHeight - 1));
}

int2 ClampMotionCoord(int2 p)
{
    return int2(clamp(p.x, 0, MotionWidth - 1), clamp(p.y, 0, MotionHeight - 1));
}

int2 ClampDepthCoord(int2 p)
{
    return int2(clamp(p.x, 0, DepthWidth - 1), clamp(p.y, 0, DepthHeight - 1));
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
    float r = SafeLoadDepthLinearFromOutputPixel(p + int2(1, 0));
    float l = SafeLoadDepthLinearFromOutputPixel(p + int2(-1, 0));
    float u = SafeLoadDepthLinearFromOutputPixel(p + int2(0, 1));
    float d = SafeLoadDepthLinearFromOutputPixel(p + int2(0, -1));

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

    residual /= max(abs(centerDepth), 1e-4);

    // More dead-zone before rejection starts.
    residual = max(residual - DepthBias - 1e-5, 0.0);

    // Softer falloff.
    float w = saturate(1.0 - residual * DepthScale);

    // Do not fully collapse taps unless it is a strong depth break.
    return lerp(0.65, 1.0, w);
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

float3 ApplyDebugTint(float3 color, float baseSharpness, float adaptiveSharpness, float edgeSharpness,
                      float finalSharpness, float distanceBoost, int debugMode)
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
    float cLuma = dot(center, float3(0.2126, 0.7152, 0.0722));
    float lumaSum = 0.0;

    float depthEdge = 1.0;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 o = kCrossOffsets[i];

        float3 tap = SafeLoadColor(p + o);
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
    float edgeSharpness = adaptiveSharpness * lerp(0.65, 1.0, edgeFactor);

    float distanceBoost = DistanceSharpnessBoost(centerDepth);
    float motionStability = saturate(adaptiveSharpness / max(Sharpness, 1e-4));
    distanceBoost = lerp(1.0, distanceBoost, motionStability);

    float boostedSharpness = edgeSharpness * distanceBoost;

    float lumaRange = ComputeLocalLumaRange(p, dot(c, float3(0.2126, 0.7152, 0.0722)));

    float unstable = saturate((lumaRange - 0.12) * 4.0);
    unstable *= unstable;

    boostedSharpness *= lerp(1.0, 0.9, unstable);
    float finalSharpness = min(boostedSharpness, 2.0);

    float3 e = c;

    // RCAS 4-neighbor pattern
    float3 bRaw = SafeLoadColor(p + int2(0, -1));
    float3 dRaw = SafeLoadColor(p + int2(-1, 0));
    float3 fRaw = SafeLoadColor(p + int2(1, 0));
    float3 hRaw = SafeLoadColor(p + int2(0, 1));
                
    // Normalize RCAS into a local 0..1-ish range for HDR/pre-tonemap input.
    // Use the original taps before depth rejection so the scale reflects the
    // true local neighborhood.
    float localScale = max(
        max(e.r, max(e.g, e.b)),
        max(
            max(bRaw.r, max(bRaw.g, bRaw.b)),
            max(
                max(dRaw.r, max(dRaw.g, dRaw.b)),
                max(
                    max(fRaw.r, max(fRaw.g, fRaw.b)),
                    max(hRaw.r, max(hRaw.g, hRaw.b))
                )
            )
        )
    );

    localScale = max(localScale, 1e-4);

    // Depth weights for cross taps.
    // crossDepths order matches kCrossOffsets:
    // 0 = up, 1 = left, 2 = right, 3 = down
    float wb = DepthWeightGrad(centerDepth, crossDepths[0], depthGrad, int2(0, -1));
    float wd = DepthWeightGrad(centerDepth, crossDepths[1], depthGrad, int2(-1, 0));
    float wf = DepthWeightGrad(centerDepth, crossDepths[2], depthGrad, int2(1, 0));
    float wh = DepthWeightGrad(centerDepth, crossDepths[3], depthGrad, int2(0, 1));

    // Prevent RCAS from pulling color across depth discontinuities.
    // Unsafe neighbors are blended back toward center.
    float3 b = lerp(e, bRaw, wb);
    float3 d = lerp(e, dRaw, wd);
    float3 f = lerp(e, fRaw, wf);
    float3 h = lerp(e, hRaw, wh);

    float3 en = e / localScale;
    float3 bn = b / localScale;
    float3 dn = d / localScale;
    float3 fn = f / localScale;
    float3 hn = h / localScale;

    // RCAS min/max ring
    float3 minRGB = min(min(bn, dn), min(fn, hn));
    float3 maxRGB = max(max(bn, dn), max(fn, hn));

    float2 peakC = float2(1.0, -4.0);

    // limiter
    float3 hitMin = minRGB / max(4.0 * maxRGB, 1e-5);
    float3 hitMax = (peakC.xxx - maxRGB) / max(4.0 * minRGB + peakC.yyy, -1e-5);

    float3 lobeRGB = max(-hitMin, hitMax);

    // RCAS is happier with roughly 0..1 range.
    float rcasSharpness = saturate(finalSharpness);
    float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * rcasSharpness;
    float rcpL = rcp(4.0 * lobe + 1.0);

    float3 output = (((bn + dn + fn + hn) * lobe + en) * rcpL) * localScale;
    
    if (Debug > 0)
        output = ApplyDebugTint(output, Sharpness, adaptiveSharpness, edgeSharpness, finalSharpness, distanceBoost, Debug);

    if (ClampOutput > 0)
        output = saturate(output);

    Dest[p] = float4(output, 1.0);
}
