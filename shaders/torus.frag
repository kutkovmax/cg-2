#version 450

// Входные интерполированные данные из вершинного шейдера
layout(location = 0) in vec3 fragNormal; // Вектор нормали в текущей точке
layout(location = 1) in vec3 fragPos;    // Координаты текущей точки в мировом пространстве
layout(location = 2) in vec3 fragColor;  // Интерполированный цвет вершины

// Выходной цвет пикселя на экране
layout(location = 0) out vec4 outColor;

// Структуры для источников освещения
struct DirectionalLight {
    vec4 direction; // xyz: вектор направления, w: флаг активности (1.0 = включен)
    vec4 color;     // rgb: цвет источника, w: интенсивность
};

struct PointLight {
    vec4 position;  // xyz: координаты источника в мире, w: не используется
    vec4 color;     // rgb: цвет источника, w: интенсивность
    vec4 params;    // x: const, y: linear, z: quadratic затухание, w: флаг активности
};

struct SpotLight {
    vec4 position;  // xyz: координаты источника в мире, w: не используется
    vec4 direction; // xyz: нормализованный вектор направления луча, w: не используется
    vec4 color;     // rgb: цвет источника, w: интенсивность
    vec4 params;    // x: cos(внутренний угол), y: cos(внешний угол), z: квадратичное затухание, w: флаг активности
};

// Набор дескрипторов 0 (Set 0): Общие данные сцены и массивы источников света
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;                // Матрица вида
    mat4 proj;                // Матрица проекции
    mat4 viewProj;            // View-Projection матрица
    vec4 cameraPos;           // Положение камеры в мировом пространстве
    vec4 ambientLight;        // rgb: цвет фона, w: интенсивность фонового освещения
    DirectionalLight dirLight;// Направленный свет
    int numPointLights;       // Число активных точечных источников
    int numSpotLights;        // Число активных прожекторов
    int pad0;
    int pad1;
} scene;

// Storage-буферы (SSBO) для динамических массивов источников без фиксированного лимита в шейдере
layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

// Набор дескрипторов 1 (Set 1): Данные конкретной модели и параметры материала
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;        // Матрица трансформации модели
    mat4 normalMatrix; // Матрица нормалей
    vec4 albedo;       // rgb: цвет альбедо (диффузный цвет)
    vec4 specular;     // rgb: цвет зеркального блика, w: коэффициент блеска (shininess)
    vec4 emissive;     // rgb: цвет самосвечения, w: флаг unlit (1.0 = рисовать чистым цветом без освещения)
} model;

void main() {
    // 1. Проверка на unlit-фигуру:
    // Маркеры источников света (сферы и конусы) рисуются чистым цветом без освещения по Блинн-Фонгу
    if (model.emissive.w > 0.5) {
        outColor = vec4(model.emissive.rgb, 1.0);
        return;
    }

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragPos); // Вектор направления взгляда (на камеру)

    // Определение цвета альбедо: умножаем на цвет вершины
    vec3 albedoColor = model.albedo.rgb * fragColor;
    vec3 specColor   = model.specular.rgb;
    float shininess  = max(model.specular.w, 1.0); // Показатель степени для остроты блика

    // 2. Фоновое освещение (Ambient Light)
    // Объект никогда не бывает абсолютно черным в тени
    vec3 ambient = scene.ambientLight.rgb * scene.ambientLight.w * albedoColor;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    // 3. Направленный источник света (Directional Light)
    if (scene.dirLight.direction.w > 0.5) {
        vec3 L = normalize(-scene.dirLight.direction.xyz); // Вектор НА источник света
        float NdotL = max(dot(N, L), 0.0);

        // Модель Блинна: используем вектор полупути (Half-way vector) между светом и взглядом
        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float specFactor = (NdotL > 0.0) ? pow(NdotH, shininess) : 0.0;

        vec3 lightEnergy = scene.dirLight.color.rgb * scene.dirLight.color.w;
        diffuse += NdotL * albedoColor * lightEnergy;
        specular += specFactor * specColor * lightEnergy;
    }

    // 4. Точечные источники света (Point Lights из SSBO)
    for (int i = 0; i < scene.numPointLights; ++i) {
        if (pointLights[i].params.w < 0.5) continue; // Источник выключен

        vec3 lightVec = pointLights[i].position.xyz - fragPos;
        float dist = length(lightVec);
        if (dist < 0.0001) continue;
        vec3 L = lightVec / dist; // Нормализованный вектор на источник

        // Затухание света по закону обратных квадратов (Inverse-Square Law):
        // Attenuation = 1.0 / (const + linear * dist + quadratic * dist^2)
        float c = max(pointLights[i].params.x, 0.0);
        float l = max(pointLights[i].params.y, 0.0);
        float q = max(pointLights[i].params.z, 0.0);
        float attenuation = 1.0 / max(c + l * dist + q * dist * dist, 0.0001);

        // Диффузная составляющая
        float NdotL = max(dot(N, L), 0.0);

        // Зеркальный блик Блинна
        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float specFactor = (NdotL > 0.0) ? pow(NdotH, shininess) : 0.0;

        vec3 lightEnergy = pointLights[i].color.rgb * pointLights[i].color.w * attenuation;
        diffuse += NdotL * albedoColor * lightEnergy;
        specular += specFactor * specColor * lightEnergy;
    }

    // 5. Прожекторные источники света (Spot Lights из SSBO)
    for (int i = 0; i < scene.numSpotLights; ++i) {
        if (spotLights[i].params.w < 0.5) continue; // Прожектор выключен

        vec3 lightVec = spotLights[i].position.xyz - fragPos;
        float dist = length(lightVec);
        if (dist < 0.0001) continue;
        vec3 L = lightVec / dist;

        // Косинус угла между осью прожектора и направлением на текущий пиксель
        vec3 spotDir = normalize(spotLights[i].direction.xyz);
        float cosTheta = dot(-L, spotDir);

        float cosInner = spotLights[i].params.x; // Косинус внутреннего угла
        float cosOuter = spotLights[i].params.y; // Косинус внешнего угла

        // Мягкие края светового конуса (Soft Cutoff):
        // Если угол внутри внутреннего конуса -> интенсивность 1.0
        // Если снаружи внешнего конуса -> интенсивность 0.0
        // Между ними -> плавный переход (спад интенсивности)
        float spotIntensity = clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 0.0001), 0.0, 1.0);
        if (spotIntensity <= 0.0) continue;

        // Квадратичное затухание по расстоянию
        float q = max(spotLights[i].params.z, 0.001);
        float attenuation = 1.0 / (1.0 + q * dist * dist);

        float NdotL = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float specFactor = (NdotL > 0.0) ? pow(NdotH, shininess) : 0.0;

        vec3 lightEnergy = spotLights[i].color.rgb * spotLights[i].color.w * attenuation * spotIntensity;
        diffuse += NdotL * albedoColor * lightEnergy;
        specular += specFactor * specColor * lightEnergy;
    }

    // Итоговый цвет пикселя: фоновый + рассеянный + блики + свечение
    vec3 finalColor = ambient + diffuse + specular + model.emissive.rgb;
    outColor = vec4(finalColor, 1.0);
}
