#include "inference.h"

#include <base/log.h>
#include <base/system.h>

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <mutex>
#include <random>

#ifdef CONF_RL_ONNX
// Disable exceptions in ONNX Runtime C++ API
#define ORT_NO_EXCEPTIONS
#include <onnxruntime_cxx_api.h>
#endif

namespace rl {

// ONNX Runtime object storage
struct CInferenceEngine::SOrtObjects
{
#ifdef CONF_RL_ONNX
	Ort::Env *m_pEnv = nullptr;
	Ort::SessionOptions *m_pSessionOptions = nullptr;
	Ort::Session *m_pSession = nullptr;
	Ort::MemoryInfo *m_pMemoryInfo = nullptr;
	std::mutex m_SessionMutex; // Protects session during reload

	~SOrtObjects()
	{
		delete m_pSession;
		delete m_pMemoryInfo;
		delete m_pSessionOptions;
		delete m_pEnv;
	}
#endif
};

CInferenceEngine::CInferenceEngine() :
	m_pOrt(std::make_unique<SOrtObjects>())
{
}

CInferenceEngine::~CInferenceEngine() = default;

CInferenceEngine::CInferenceEngine(CInferenceEngine &&) noexcept = default;
CInferenceEngine &CInferenceEngine::operator=(CInferenceEngine &&) noexcept = default;

bool CInferenceEngine::Init(const SInferenceConfig &Config)
{
#ifndef CONF_RL_ONNX
	SetLastError("ONNX Runtime support not compiled in (CONF_RL_ONNX not defined)");
	return false;
#else
	m_Config = Config;

	// Create ONNX Runtime environment
	m_pOrt->m_pEnv = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DDNet-RL");

	// Create session options
	m_pOrt->m_pSessionOptions = new Ort::SessionOptions();

	// Set thread counts
	m_pOrt->m_pSessionOptions->SetIntraOpNumThreads(Config.m_IntraOpNumThreads);
	m_pOrt->m_pSessionOptions->SetInterOpNumThreads(Config.m_InterOpNumThreads);

	// Enable optimizations
	m_pOrt->m_pSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	// Configure execution provider
	if(Config.m_Provider == EExecutionProvider::CUDA)
	{
		// Try to use CUDA provider (simple API with device_id only)
		OrtStatus *pStatus = OrtSessionOptionsAppendExecutionProvider_CUDA(
			static_cast<OrtSessionOptions *>(*m_pOrt->m_pSessionOptions), Config.m_DeviceId);
		if(pStatus != nullptr)
		{
			const char *pError = Ort::GetApi().GetErrorMessage(pStatus);
			log_warn("rl/inference", "CUDA provider failed: %s, falling back to CPU", pError);
			Ort::GetApi().ReleaseStatus(pStatus);
		}
		else
		{
			log_info("rl/inference", "Using CUDA execution provider (device %d)", Config.m_DeviceId);
		}
	}
	else
	{
		log_info("rl/inference", "Using CPU execution provider");
	}

	// Create memory info for tensor allocation
	m_pOrt->m_pMemoryInfo = new Ort::MemoryInfo(
		Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

	log_info("rl/inference", "ONNX Runtime initialized successfully");
	return true;
#endif
}

bool CInferenceEngine::LoadModel(const char *pPath)
{
#ifndef CONF_RL_ONNX
	SetLastError("ONNX Runtime support not compiled in");
	return false;
#else
	if(!m_pOrt->m_pEnv)
	{
		SetLastError("Engine not initialized. Call Init() first.");
		return false;
	}

	m_ModelPath = pPath;

	std::lock_guard<std::mutex> Lock(m_pOrt->m_SessionMutex);

	// Delete old session
	delete m_pOrt->m_pSession;
	m_pOrt->m_pSession = nullptr;
	m_pSession = nullptr;

	// Create session from model file
	m_pOrt->m_pSession = new Ort::Session(*m_pOrt->m_pEnv, pPath, *m_pOrt->m_pSessionOptions);
	m_pSession = m_pOrt->m_pSession;

	// Cache input/output names
	CacheIONames();

	// Validate model structure
	if(m_InputNamesStorage.empty() || m_OutputNamesStorage.empty())
	{
		SetLastError("Model has no inputs or outputs");
		delete m_pOrt->m_pSession;
		m_pOrt->m_pSession = nullptr;
		m_pSession = nullptr;
		return false;
	}

	// Get input shape to determine observation size
	Ort::TypeInfo InputTypeInfo = m_pSession->GetInputTypeInfo(0);
	auto InputTensorInfo = InputTypeInfo.GetTensorTypeAndShapeInfo();
	auto InputShape = InputTensorInfo.GetShape();

	if(InputShape.size() >= 2)
	{
		// Shape is [batch, obs_size] - obs_size might be -1 for dynamic
		m_ObservationSize = (InputShape[1] > 0) ? static_cast<int>(InputShape[1]) : SObservation::GetFloatCount();
	}
	else
	{
		m_ObservationSize = SObservation::GetFloatCount();
	}

	// Get output shape
	Ort::TypeInfo OutputTypeInfo = m_pSession->GetOutputTypeInfo(0);
	auto OutputTensorInfo = OutputTypeInfo.GetTensorTypeAndShapeInfo();
	auto OutputShape = OutputTensorInfo.GetShape();

	if(OutputShape.size() >= 2)
	{
		m_OutputSize = (OutputShape[1] > 0) ? static_cast<int>(OutputShape[1]) : SNetworkOutput::GetFloatCount();
	}
	else
	{
		m_OutputSize = SNetworkOutput::GetFloatCount();
	}

	m_ModelVersion++;

	log_info("rl/inference", "Model loaded: %s (obs_size=%d, out_size=%d, version=%u)",
		pPath, m_ObservationSize, m_OutputSize, m_ModelVersion);

	return true;
#endif
}

bool CInferenceEngine::Infer(const float *pObservations, int NumBots, float *pOutputs)
{
#ifndef CONF_RL_ONNX
	SetLastError("ONNX Runtime support not compiled in");
	return false;
#else
	if(!m_pSession)
	{
		SetLastError("No model loaded");
		return false;
	}

	if(NumBots <= 0)
	{
		return true; // Nothing to do
	}

	std::lock_guard<std::mutex> Lock(m_pOrt->m_SessionMutex);

	// Create input tensor
	std::array<int64_t, 2> InputShape = {NumBots, m_ObservationSize};
	Ort::Value InputTensor = Ort::Value::CreateTensor<float>(
		*m_pOrt->m_pMemoryInfo,
		const_cast<float *>(pObservations),
		NumBots * m_ObservationSize,
		InputShape.data(),
		InputShape.size());

	// Run inference
	auto OutputTensors = m_pSession->Run(
		Ort::RunOptions{nullptr},
		m_InputNames.data(),
		&InputTensor,
		1,
		m_OutputNames.data(),
		m_OutputNames.size());

	// Copy output
	if(!OutputTensors.empty() && OutputTensors[0].IsTensor())
	{
		const float *pOutputData = OutputTensors[0].GetTensorData<float>();
		mem_copy(pOutputs, pOutputData, NumBots * m_OutputSize * sizeof(float));
	}

	return true;
#endif
}

bool CInferenceEngine::InferActions(const float *pObservations, int NumBots,
	SAction *pActions, float *pLogProbs, float *pValues, bool Deterministic)
{
	if(NumBots <= 0)
		return true;

	// Allocate output buffer if needed
	const int TotalOutputSize = NumBots * m_OutputSize;
	if(static_cast<int>(m_OutputBuffer.size()) < TotalOutputSize)
	{
		m_OutputBuffer.resize(TotalOutputSize);
	}

	// Run inference
	if(!Infer(pObservations, NumBots, m_OutputBuffer.data()))
	{
		return false;
	}

	// Convert network outputs to actions
	thread_local std::mt19937 s_Rng(std::random_device{}());

	for(int i = 0; i < NumBots; i++)
	{
		const float *pOut = &m_OutputBuffer[i * m_OutputSize];
		SAction &Action = pActions[i];

		float LogProb = 0.0f;

		// Direction (3-way categorical)
		if(Deterministic)
		{
			Action.m_Direction = ArgmaxCategorical(pOut, 3) - 1; // Convert [0,1,2] to [-1,0,1]
		}
		else
		{
			float lp;
			Action.m_Direction = SampleCategorical(pOut, 3, &lp) - 1;
			LogProb += lp;
		}
		pOut += 3;

		// Jump (binary)
		float JumpProb = 1.0f / (1.0f + std::exp(-pOut[0])); // Sigmoid
		if(Deterministic)
		{
			Action.m_Jump = JumpProb > 0.5f ? 1 : 0;
		}
		else
		{
			std::bernoulli_distribution JumpDist(JumpProb);
			Action.m_Jump = JumpDist(s_Rng) ? 1 : 0;
		}
		LogProb += Action.m_Jump ? std::log(JumpProb + 1e-8f) : std::log(1.0f - JumpProb + 1e-8f);
		pOut += 1;

		// Hook (binary)
		float HookProb = 1.0f / (1.0f + std::exp(-pOut[0]));
		if(Deterministic)
		{
			Action.m_Hook = HookProb > 0.5f ? 1 : 0;
		}
		else
		{
			std::bernoulli_distribution HookDist(HookProb);
			Action.m_Hook = HookDist(s_Rng) ? 1 : 0;
		}
		LogProb += Action.m_Hook ? std::log(HookProb + 1e-8f) : std::log(1.0f - HookProb + 1e-8f);
		pOut += 1;

		// Fire (binary)
		float FireProb = 1.0f / (1.0f + std::exp(-pOut[0]));
		if(Deterministic)
		{
			Action.m_Fire = FireProb > 0.5f ? 1 : 0;
		}
		else
		{
			std::bernoulli_distribution FireDist(FireProb);
			Action.m_Fire = FireDist(s_Rng) ? 1 : 0;
		}
		LogProb += Action.m_Fire ? std::log(FireProb + 1e-8f) : std::log(1.0f - FireProb + 1e-8f);
		pOut += 1;

		// Weapon (3-way categorical: prev, none, next)
		if(Deterministic)
		{
			Action.m_WeaponNext = ArgmaxCategorical(pOut, 3) - 1;
		}
		else
		{
			float lp;
			Action.m_WeaponNext = SampleCategorical(pOut, 3, &lp) - 1;
			LogProb += lp;
		}
		pOut += 3;

		// Aim (continuous Gaussian)
		float AimMeanX = pOut[0];
		float AimMeanY = pOut[1];
		float AimLogStdX = pOut[2];
		float AimLogStdY = pOut[3];

		if(Deterministic)
		{
			Action.m_AimX = AimMeanX;
			Action.m_AimY = AimMeanY;
		}
		else
		{
			float lpX, lpY;
			Action.m_AimX = SampleGaussian(AimMeanX, AimLogStdX, &lpX);
			Action.m_AimY = SampleGaussian(AimMeanY, AimLogStdY, &lpY);
			LogProb += lpX + lpY;
		}
		pOut += 4;

		// Value estimate
		float Value = pOut[0];

		// Store log probs and values if requested
		if(pLogProbs)
		{
			pLogProbs[i] = LogProb;
		}
		if(pValues)
		{
			pValues[i] = Value;
		}
	}

	return true;
}

bool CInferenceEngine::ReloadModel()
{
	if(m_ModelPath.empty())
	{
		SetLastError("No model path set");
		return false;
	}

	return LoadModel(m_ModelPath.c_str());
}

bool CInferenceEngine::CheckAndReloadModel(SModelReloadSignal *pSignal)
{
	if(!pSignal)
		return false;

	// Check if new model is ready (atomic read)
	if(__atomic_load_n(&pSignal->m_NewModelReady, __ATOMIC_ACQUIRE) == 0)
	{
		return false;
	}

	// Load the new model
	bool Success = LoadModel(pSignal->m_ModelPath);

	if(Success)
	{
		// Update version and signal that we've loaded
		pSignal->m_ModelVersion = m_ModelVersion;
		__atomic_store_n(&pSignal->m_ModelLoaded, 1, __ATOMIC_RELEASE);
	}

	// Clear the ready signal
	__atomic_store_n(&pSignal->m_NewModelReady, 0, __ATOMIC_RELEASE);

	return Success;
}

void CInferenceEngine::CacheIONames()
{
#ifdef CONF_RL_ONNX
	if(!m_pSession)
		return;

	Ort::AllocatorWithDefaultOptions Allocator;

	// Get input names
	size_t NumInputs = m_pSession->GetInputCount();
	m_InputNamesStorage.resize(NumInputs);
	m_InputNames.resize(NumInputs);

	for(size_t i = 0; i < NumInputs; i++)
	{
		auto Name = m_pSession->GetInputNameAllocated(i, Allocator);
		m_InputNamesStorage[i] = Name.get();
		m_InputNames[i] = m_InputNamesStorage[i].c_str();
	}

	// Get output names
	size_t NumOutputs = m_pSession->GetOutputCount();
	m_OutputNamesStorage.resize(NumOutputs);
	m_OutputNames.resize(NumOutputs);

	for(size_t i = 0; i < NumOutputs; i++)
	{
		auto Name = m_pSession->GetOutputNameAllocated(i, Allocator);
		m_OutputNamesStorage[i] = Name.get();
		m_OutputNames[i] = m_OutputNamesStorage[i].c_str();
	}
#endif
}

void CInferenceEngine::SetLastError(const char *pFormat, ...)
{
	char aBuffer[1024];
	va_list Args;
	va_start(Args, pFormat);
	str_format_v(aBuffer, sizeof(aBuffer), pFormat, Args);
	va_end(Args);

	m_LastError = aBuffer;
	log_error("rl/inference", "%s", aBuffer);
}

int CInferenceEngine::SampleCategorical(const float *pLogits, int NumCategories, float *pLogProb)
{
	thread_local std::mt19937 s_Rng(std::random_device{}());

	// Compute softmax probabilities
	float MaxLogit = pLogits[0];
	for(int i = 1; i < NumCategories; i++)
	{
		if(pLogits[i] > MaxLogit)
			MaxLogit = pLogits[i];
	}

	std::vector<float> Probs(NumCategories);
	float Sum = 0.0f;
	for(int i = 0; i < NumCategories; i++)
	{
		Probs[i] = std::exp(pLogits[i] - MaxLogit);
		Sum += Probs[i];
	}

	for(int i = 0; i < NumCategories; i++)
	{
		Probs[i] /= Sum;
	}

	// Sample
	std::discrete_distribution<int> Dist(Probs.begin(), Probs.end());
	int Sample = Dist(s_Rng);

	if(pLogProb)
	{
		*pLogProb = std::log(Probs[Sample] + 1e-8f);
	}

	return Sample;
}

float CInferenceEngine::SampleGaussian(float Mean, float LogStd, float *pLogProb)
{
	thread_local std::mt19937 s_Rng(std::random_device{}());

	float Std = std::exp(LogStd);
	std::normal_distribution<float> Dist(Mean, Std);
	float Sample = Dist(s_Rng);

	if(pLogProb)
	{
		// Log probability of Gaussian
		constexpr float LOG_2PI = 1.8378770664093453f;
		float Diff = Sample - Mean;
		*pLogProb = -0.5f * (LOG_2PI + 2.0f * LogStd + (Diff * Diff) / (Std * Std));
	}

	return Sample;
}

int CInferenceEngine::ArgmaxCategorical(const float *pLogits, int NumCategories)
{
	int MaxIdx = 0;
	float MaxVal = pLogits[0];
	for(int i = 1; i < NumCategories; i++)
	{
		if(pLogits[i] > MaxVal)
		{
			MaxVal = pLogits[i];
			MaxIdx = i;
		}
	}
	return MaxIdx;
}

// Global inference engine
static std::unique_ptr<CInferenceEngine> s_pGlobalEngine;

CInferenceEngine *GetInferenceEngine()
{
	return s_pGlobalEngine.get();
}

bool InitGlobalInferenceEngine(const SInferenceConfig &Config, const char *pModelPath)
{
	s_pGlobalEngine = std::make_unique<CInferenceEngine>();

	if(!s_pGlobalEngine->Init(Config))
	{
		s_pGlobalEngine.reset();
		return false;
	}

	if(pModelPath && pModelPath[0])
	{
		if(!s_pGlobalEngine->LoadModel(pModelPath))
		{
			s_pGlobalEngine.reset();
			return false;
		}
	}

	return true;
}

void ShutdownGlobalInferenceEngine()
{
	s_pGlobalEngine.reset();
}

} // namespace rl
