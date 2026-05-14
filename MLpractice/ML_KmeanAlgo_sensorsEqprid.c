#include <stdio.h>
#include <math.h>
#include <float.h>

#define NUM_FEATURES 5
#define NUM_CLUSTERS 3

typedef enum
{
    HEALTH_NORMAL = 0,
    HEALTH_WARNING,
    HEALTH_FAULT_RISK
} MachineHealth_t;

typedef struct
{
    float rpm;
    float temperature;
    float vibration;
    float current;
    float quality_score;
} MachineData_t;

typedef struct
{
    float feature[NUM_FEATURES];
} FeatureVector_t;

/*
    Feature order:
    feature[0] = rpm
    feature[1] = temperature
    feature[2] = vibration
    feature[3] = current
    feature[4] = quality_score
*/

/*
    These min/max values should come from training data.
    Example values only.
*/
static const float feature_min[NUM_FEATURES] =
{
    850.0f,  // rpm min
    60.0f,   // temperature min
    1.5f,    // vibration min
    10.0f,   // current min
    50.0f    // quality score min
};

static const float feature_max[NUM_FEATURES] =
{
    1600.0f, // rpm max
    105.0f,  // temperature max
    9.5f,    // vibration max
    28.0f,   // current max
    100.0f   // quality score max
};

/*
    Example centroids after normalization.

    Cluster 0 = Normal
    Cluster 1 = Warning
    Cluster 2 = Fault Risk

    These are example values.
    In real project, copy final centroids from Python training output.
*/
static const float centroids[NUM_CLUSTERS][NUM_FEATURES] =
{
    // rpm,  temp, vib,  current, quality
    {0.45f, 0.20f, 0.15f, 0.20f, 0.90f},  // Normal
    {0.60f, 0.55f, 0.50f, 0.55f, 0.65f},  // Warning
    {0.30f, 0.85f, 0.85f, 0.85f, 0.25f}   // Fault Risk
};

static const MachineHealth_t cluster_health_map[NUM_CLUSTERS] =
{
    HEALTH_NORMAL,
    HEALTH_WARNING,
    HEALTH_FAULT_RISK
};

static float normalize_value(float value, float min_value, float max_value)
{
    if ((max_value - min_value) == 0.0f)
    {
        return 0.0f;
    }

    float normalized = (value - min_value) / (max_value - min_value);

    if (normalized < 0.0f)
    {
        normalized = 0.0f;
    }

    if (normalized > 1.0f)
    {
        normalized = 1.0f;
    }

    return normalized;
}

static void convert_to_feature_vector(
    const MachineData_t *data,
    FeatureVector_t *vector
)
{
    vector->feature[0] = normalize_value(
        data->rpm,
        feature_min[0],
        feature_max[0]
    );

    vector->feature[1] = normalize_value(
        data->temperature,
        feature_min[1],
        feature_max[1]
    );

    vector->feature[2] = normalize_value(
        data->vibration,
        feature_min[2],
        feature_max[2]
    );

    vector->feature[3] = normalize_value(
        data->current,
        feature_min[3],
        feature_max[3]
    );

    vector->feature[4] = normalize_value(
        data->quality_score,
        feature_min[4],
        feature_max[4]
    );
}

static float calculate_squared_distance(
    const FeatureVector_t *vector,
    const float centroid[NUM_FEATURES]
)
{
    float distance = 0.0f;

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        float diff = vector->feature[i] - centroid[i];
        distance += diff * diff;
    }

    return distance;
}

static int predict_cluster(const MachineData_t *data)
{
    FeatureVector_t vector;
    convert_to_feature_vector(data, &vector);

    int nearest_cluster = 0;
    float min_distance = FLT_MAX;

    for (int cluster = 0; cluster < NUM_CLUSTERS; cluster++)
    {
        float distance = calculate_squared_distance(
            &vector,
            centroids[cluster]
        );

        if (distance < min_distance)
        {
            min_distance = distance;
            nearest_cluster = cluster;
        }
    }

    return nearest_cluster;
}

static MachineHealth_t predict_health(const MachineData_t *data)
{
    int cluster = predict_cluster(data);
    return cluster_health_map[cluster];
}

static const char *health_to_string(MachineHealth_t health)
{
    switch (health)
    {
        case HEALTH_NORMAL:
            return "NORMAL";

        case HEALTH_WARNING:
            return "WARNING";

        case HEALTH_FAULT_RISK:
            return "FAULT_RISK";

        default:
            return "UNKNOWN";
    }
}

static const char *action_from_health(MachineHealth_t health)
{
    switch (health)
    {
        case HEALTH_NORMAL:
            return "Continue running";

        case HEALTH_WARNING:
            return "Schedule inspection";

        case HEALTH_FAULT_RISK:
            return "Stop machine and inspect immediately";

        default:
            return "Unknown action";
    }
}

int main(void)
{
    MachineData_t machine = {
        .rpm = 980.0f,
        .temperature = 96.0f,
        .vibration = 8.4f,
        .current = 25.0f,
        .quality_score = 62.0f
    };

    int cluster = predict_cluster(&machine);
    MachineHealth_t health = predict_health(&machine);

    printf("Machine Sensor Data:\n");
    printf("RPM: %.2f\n", machine.rpm);
    printf("Temperature: %.2f\n", machine.temperature);
    printf("Vibration: %.2f\n", machine.vibration);
    printf("Current: %.2f\n", machine.current);
    printf("Quality Score: %.2f\n", machine.quality_score);

    printf("\nPredicted Cluster: %d\n", cluster);
    printf("Machine Health: %s\n", health_to_string(health));
    printf("Action: %s\n", action_from_health(health));

    return 0;
}
