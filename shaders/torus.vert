#version 450

// Входные атрибуты вершин из Vertex Buffer
layout(location = 0) in vec3 inPosition; // Позиция вершины (X, Y, Z) в локальных координатах
layout(location = 1) in vec3 inNormal;   // Нормаль вершины (процедурно сгенерированная)
layout(location = 2) in vec3 inColor;    // Процедурный цвет вершины (радужный градиент или белый)

// Структура направленного источника света
struct DirectionalLight {
    vec4 direction; // xyz: вектор направления света, w: включен (1.0 = вкл, 0.0 = выкл)
    vec4 color;     // rgb: цвет света, w: интенсивность
};

// Набор дескрипторов 0 (Set 0): Общие данные сцены
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;                // Матрица камеры (View Matrix)
    mat4 proj;                // Матрица проекции (Projection Matrix)
    mat4 viewProj;            // Скомбинированная матрица View-Projection
    vec4 cameraPos;           // xyz: положение камеры в мировом пространстве
    vec4 ambientLight;        // rgb: цвет фонового света, w: интенсивность фона
    DirectionalLight dirLight;// Параметры направленного источника
    int numPointLights;       // Количество активных точечных источников
    int numSpotLights;        // Количество активных прожекторов
    int pad0;                 // Выравнивание по стандарту std140
    int pad1;                 // Выравнивание по стандарту std140
} scene;

// Набор дескрипторов 1 (Set 1): Данные конкретной модели/объекта
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;        // Матрица модели (Model Matrix: Translate * Rotate * Scale)
    mat4 normalMatrix; // Матрица нормалей: transpose(inverse(model)) для сохранения перпендикулярности
    vec4 albedo;       // rgb: базовый цвет материала, w: флаг использования цвета вершин
    vec4 specular;     // rgb: цвет блика, w: показатель блеска (shininess)
    vec4 emissive;     // rgb: цвет самосвечения, w: флаг unlit (1.0 = без освещения, для маркеров света)
} model;

// Выходные данные для передачи во фрагментный шейдер
layout(location = 0) out vec3 fragNormal; // Нормаль в мировых координатах
layout(location = 1) out vec3 fragPos;    // Координаты фрагмента в мировом пространстве
layout(location = 2) out vec3 fragColor;  // Базовый цвет вершины

void main() {
    // 1. Переводим локальную позицию вершины в мировое пространство
    vec4 worldPos = model.model * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    
    // 2. Трансформируем вектор нормали матрицей normalMatrix (чтобы избежать искажений при масштабировании)
    fragNormal = normalize(mat3(model.normalMatrix) * inNormal);
    
    // 3. Передаем процедурный цвет во фрагментный шейдер
    fragColor = inColor;
    
    // 4. Финальная позиция на экране: перевод из мировых координат в clip space
    gl_Position = scene.proj * scene.view * worldPos;
}
