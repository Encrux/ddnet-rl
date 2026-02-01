#ifndef GAME_RL_INFERENCE_H
#define GAME_RL_INFERENCE_H

#include "rl_protocol.h"

#include <memory>
#include <string>
#include <vector>

// Forward declarations for ONNX Runtime types
namespace Ort {
class Env;
class Session;
class SessionOptions;
class MemoryInfo;
class Value;
} // namespace Ort

namespace rl {

enum class EExecutionProvider
{
	CPU,
	CUDA,
	// Future: TensorRT, DirectML, etc.
};

struct SInferenceConfig
{
	EExecutionProvider m_Provider = EExecutionProvider::CPU;
	int m_DeviceId = 0;                    // GPU device ID for CUDA
	int m_IntraOpNumThreads = 1;           // Threads for single operation
	int m_InterOpNumThreads = 1;           // Threads for parallel operations
	bool m_EnableMemoryArenaOnCPU = true;
	bool m_EnableMemoryArenaOnGPU = true;
	size_t m_GPUMemoryLimit = 0;           // 0 = no limit
};

class CInferenceEngine
{
public:
	CInferenceEngine();
	~CInferenceEngine();

	// Disable copy
	CInferenceEngine(const CInferenceEngine &) = delete;
	CInferenceEngine &operator=(const CInferenceEngine &) = delete;

	// Enable move
	CInferenceEngine(CInferenceEngine &&) noexcept;
	CInferenceEngine &operator=(CInferenceEngine &&) noexcept;

	/**
	 * Initialize the ONNX Runtime environment.
	 * Must be called before LoadModel.
	 *
	 * @param Config Inference configuration
	 * @return true if initialization succeeded
	 */
	bool Init(const SInferenceConfig &Config);

	/**
	 * Load an ONNX model from file.
	 *
	 * @param pPath Path to the .onnx model file
	 * @return true if model loaded successfully
	 */
	bool LoadModel(const char *pPath);

	/**
	 * Run batched inference on observations.
	 *
	 * @param pObservations Flat array of observations [NumBots * ObsSize]
	 * @param NumBots Number of bots/observations in the batch
	 * @param pOutputs Output array for network outputs [NumBots * OutputSize]
	 * @return true if inference succeeded
	 */
	bool Infer(const float *pObservations, int NumBots, float *pOutputs);

	/**
	 * Run inference and convert to actions.
	 * This samples from the policy distribution.
	 *
	 * @param pObservations Flat array of observations
	 * @param NumBots Number of bots
	 * @param pActions Output array for actions
	 * @param pLogProbs Optional output for log probabilities (for training)
	 * @param pValues Optional output for value estimates
	 * @param Deterministic If true, use argmax instead of sampling
	 * @return true if inference succeeded
	 */
	bool InferActions(const float *pObservations, int NumBots,
		SAction *pActions, float *pLogProbs = nullptr,
		float *pValues = nullptr, bool Deterministic = false);

	/**
	 * Reload the model from disk (hot reload for training).
	 * Thread-safe: blocks inference during reload.
	 *
	 * @return true if reload succeeded
	 */
	bool ReloadModel();

	/**
	 * Check if a new model is available and reload if so.
	 * Non-blocking check, only reloads if signaled.
	 *
	 * @param pModelSignal Pointer to model reload signal in shared memory
	 * @return true if model was reloaded
	 */
	bool CheckAndReloadModel(SModelReloadSignal *pModelSignal);

	/**
	 * Get the current model path.
	 */
	const char *GetModelPath() const { return m_ModelPath.c_str(); }

	/**
	 * Check if the engine is initialized and ready.
	 */
	bool IsReady() const { return m_pSession != nullptr; }

	/**
	 * Get observation size (number of floats).
	 */
	int GetObservationSize() const { return m_ObservationSize; }

	/**
	 * Get output size (number of floats).
	 */
	int GetOutputSize() const { return m_OutputSize; }

	/**
	 * Get the current model version.
	 */
	uint32_t GetModelVersion() const { return m_ModelVersion; }

	/**
	 * Get last error message.
	 */
	const char *GetLastError() const { return m_LastError.c_str(); }

private:
	// ONNX Runtime objects (using raw pointers to avoid header dependency)
	struct SOrtObjects;
	std::unique_ptr<SOrtObjects> m_pOrt;

	// Cached session pointer for quick access
	Ort::Session *m_pSession = nullptr;

	// Configuration
	SInferenceConfig m_Config;
	std::string m_ModelPath;
	std::string m_LastError;

	// Model metadata
	int m_ObservationSize = 0;
	int m_OutputSize = 0;
	uint32_t m_ModelVersion = 0;

	// Input/output names (cached from model)
	std::vector<const char *> m_InputNames;
	std::vector<const char *> m_OutputNames;
	std::vector<std::string> m_InputNamesStorage;
	std::vector<std::string> m_OutputNamesStorage;

	// Pre-allocated buffers for inference
	std::vector<float> m_OutputBuffer;

	// Helper methods
	bool CreateSession();
	void CacheIONames();
	void SetLastError(const char *pFormat, ...);

	// Action sampling helpers
	int SampleCategorical(const float *pLogits, int NumCategories, float *pLogProb);
	float SampleGaussian(float Mean, float LogStd, float *pLogProb);
	int ArgmaxCategorical(const float *pLogits, int NumCategories);
};

/**
 * Global inference engine instance for the server.
 * Initialized once, used by all bots.
 */
CInferenceEngine *GetInferenceEngine();

/**
 * Initialize the global inference engine.
 * Call this once during server startup.
 */
bool InitGlobalInferenceEngine(const SInferenceConfig &Config, const char *pModelPath);

/**
 * Shutdown the global inference engine.
 * Call this during server shutdown.
 */
void ShutdownGlobalInferenceEngine();

} // namespace rl

#endif // GAME_RL_INFERENCE_H
