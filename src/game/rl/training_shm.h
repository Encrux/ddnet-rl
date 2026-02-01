#ifndef GAME_RL_TRAINING_SHM_H
#define GAME_RL_TRAINING_SHM_H

#include "rl_protocol.h"

#include <cstddef>
#include <cstdint>

namespace rl {

// Default configuration
constexpr size_t DEFAULT_TRANSITION_BUFFER_SIZE = 65536; // 64K transitions
constexpr const char *DEFAULT_SHM_NAME = "/ddnet_rl_training";

/**
 * Shared memory manager for training data exchange.
 *
 * Layout:
 * [STrainingSharedMemory header]
 * [STransition array]
 *
 * The C++ server writes transitions to the ring buffer.
 * Python reads transitions and performs PPO updates.
 * Python writes new model path and signals for hot reload.
 */
class CTrainingSharedMemory
{
public:
	CTrainingSharedMemory();
	~CTrainingSharedMemory();

	// Disable copy
	CTrainingSharedMemory(const CTrainingSharedMemory &) = delete;
	CTrainingSharedMemory &operator=(const CTrainingSharedMemory &) = delete;

	/**
	 * Create a new shared memory region (server/owner side).
	 *
	 * @param pName Shared memory name (e.g., "/ddnet_rl_training")
	 * @param Capacity Number of transitions the buffer can hold
	 * @return true if creation succeeded
	 */
	bool Create(const char *pName, size_t Capacity = DEFAULT_TRANSITION_BUFFER_SIZE);

	/**
	 * Open an existing shared memory region (client/Python side).
	 *
	 * @param pName Shared memory name
	 * @return true if open succeeded
	 */
	bool Open(const char *pName);

	/**
	 * Close the shared memory region.
	 */
	void Close();

	/**
	 * Unlink (delete) the shared memory region.
	 * Call this on cleanup to remove the shared memory object.
	 */
	void Unlink();

	/**
	 * Check if the shared memory is open and valid.
	 */
	bool IsValid() const { return m_pMemory != nullptr; }

	/**
	 * Write a transition to the ring buffer.
	 * Non-blocking: overwrites oldest data if buffer is full.
	 *
	 * @param Transition The transition to write
	 * @return true if write succeeded
	 */
	bool WriteTransition(const STransition &Transition);

	/**
	 * Write multiple transitions to the ring buffer.
	 * More efficient than individual writes.
	 *
	 * @param pTransitions Array of transitions
	 * @param Count Number of transitions
	 * @return Number of transitions written
	 */
	size_t WriteTransitions(const STransition *pTransitions, size_t Count);

	/**
	 * Read transitions from the ring buffer.
	 * Reads up to MaxCount transitions that haven't been read yet.
	 *
	 * @param pTransitions Output array for transitions
	 * @param MaxCount Maximum number to read
	 * @return Number of transitions read
	 */
	size_t ReadTransitions(STransition *pTransitions, size_t MaxCount);

	/**
	 * Get the number of transitions available to read.
	 */
	size_t GetAvailableTransitions() const;

	/**
	 * Get the total number of transitions written.
	 */
	uint64_t GetTotalWritten() const;

	/**
	 * Get the total number of transitions read.
	 */
	uint64_t GetTotalRead() const;

	/**
	 * Get pointer to the model reload signal structure.
	 * Use this to signal model updates.
	 */
	SModelReloadSignal *GetModelSignal();

	/**
	 * Signal that a new model is ready to be loaded.
	 *
	 * @param pModelPath Path to the new ONNX model
	 */
	void SignalNewModel(const char *pModelPath);

	/**
	 * Check if the C++ side has loaded the new model.
	 */
	bool IsModelLoaded() const;

	/**
	 * Update training statistics.
	 */
	void IncrementEpisodeCount();
	void AddSteps(uint64_t Steps);
	void AddReward(double Reward);
	void SetTrainingActive(bool Active);

	/**
	 * Get training statistics.
	 */
	uint64_t GetEpisodeCount() const;
	uint64_t GetStepCount() const;
	double GetTotalReward() const;
	bool IsTrainingActive() const;

	/**
	 * Get the buffer capacity.
	 */
	size_t GetCapacity() const;

	/**
	 * Get the shared memory name.
	 */
	const char *GetName() const { return m_Name; }

private:
	void *m_pMemory = nullptr;
	size_t m_MemorySize = 0;
	int m_Fd = -1;
	bool m_IsOwner = false;
	char m_Name[256] = {0};

	// Cached pointers into shared memory
	STrainingSharedMemory *m_pHeader = nullptr;
	STransition *m_pTransitions = nullptr;

	// Calculate total shared memory size
	static size_t CalculateMemorySize(size_t Capacity);

	// Get transition at index (with wrap-around)
	STransition *GetTransition(size_t Index);
	const STransition *GetTransition(size_t Index) const;
};

/**
 * RAII wrapper for shared memory that automatically unlinks on destruction.
 */
class CTrainingSharedMemoryOwner
{
public:
	CTrainingSharedMemoryOwner() = default;
	~CTrainingSharedMemoryOwner()
	{
		if(m_Shm.IsValid())
		{
			m_Shm.Unlink();
		}
	}

	CTrainingSharedMemory &Get() { return m_Shm; }
	const CTrainingSharedMemory &Get() const { return m_Shm; }

	bool Create(const char *pName, size_t Capacity = DEFAULT_TRANSITION_BUFFER_SIZE)
	{
		return m_Shm.Create(pName, Capacity);
	}

private:
	CTrainingSharedMemory m_Shm;
};

} // namespace rl

#endif // GAME_RL_TRAINING_SHM_H
