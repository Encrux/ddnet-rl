#include "training_shm.h"

#include <base/log.h>
#include <base/system.h>

#include <cstring>

#ifdef CONF_FAMILY_UNIX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef CONF_FAMILY_WINDOWS
#include <windows.h>
#endif

namespace rl {

CTrainingSharedMemory::CTrainingSharedMemory() = default;

CTrainingSharedMemory::~CTrainingSharedMemory()
{
	Close();
}

size_t CTrainingSharedMemory::CalculateMemorySize(size_t Capacity)
{
	return sizeof(STrainingSharedMemory) + Capacity * sizeof(STransition);
}

bool CTrainingSharedMemory::Create(const char *pName, size_t Capacity)
{
	if(m_pMemory)
	{
		log_error("rl/shm", "Shared memory already initialized");
		return false;
	}

	str_copy(m_Name, pName);
	m_MemorySize = CalculateMemorySize(Capacity);
	m_IsOwner = true;

#ifdef CONF_FAMILY_UNIX
	// Remove existing shared memory if it exists
	shm_unlink(pName);

	// Create shared memory object
	m_Fd = shm_open(pName, O_CREAT | O_RDWR | O_EXCL, 0666);
	if(m_Fd == -1)
	{
		log_error("rl/shm", "Failed to create shared memory '%s': %s", pName, strerror(errno));
		return false;
	}

	// Set size
	if(ftruncate(m_Fd, m_MemorySize) == -1)
	{
		log_error("rl/shm", "Failed to set shared memory size: %s", strerror(errno));
		close(m_Fd);
		shm_unlink(pName);
		m_Fd = -1;
		return false;
	}

	// Map into address space
	m_pMemory = mmap(nullptr, m_MemorySize, PROT_READ | PROT_WRITE, MAP_SHARED, m_Fd, 0);
	if(m_pMemory == MAP_FAILED)
	{
		log_error("rl/shm", "Failed to map shared memory: %s", strerror(errno));
		close(m_Fd);
		shm_unlink(pName);
		m_pMemory = nullptr;
		m_Fd = -1;
		return false;
	}
#endif

#ifdef CONF_FAMILY_WINDOWS
	// Create file mapping
	HANDLE hMapFile = CreateFileMappingA(
		INVALID_HANDLE_VALUE,
		nullptr,
		PAGE_READWRITE,
		(DWORD)(m_MemorySize >> 32),
		(DWORD)(m_MemorySize & 0xFFFFFFFF),
		pName);

	if(hMapFile == nullptr)
	{
		log_error("rl/shm", "Failed to create file mapping '%s': %lu", pName, GetLastError());
		return false;
	}

	m_Fd = (int)(intptr_t)hMapFile;

	// Map view
	m_pMemory = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, m_MemorySize);
	if(m_pMemory == nullptr)
	{
		log_error("rl/shm", "Failed to map view of file: %lu", GetLastError());
		CloseHandle(hMapFile);
		m_Fd = -1;
		return false;
	}
#endif

	// Initialize the header
	mem_zero(m_pMemory, m_MemorySize);
	m_pHeader = static_cast<STrainingSharedMemory *>(m_pMemory);
	m_pHeader->m_Header.m_Capacity = static_cast<uint32_t>(Capacity);
	m_pHeader->m_Header.m_TransitionSize = sizeof(STransition);
	m_pHeader->m_Header.m_WriteIndex = 0;
	m_pHeader->m_Header.m_ReadIndex = 0;
	m_pHeader->m_Header.m_TotalWritten = 0;
	m_pHeader->m_Header.m_TotalRead = 0;
	m_pHeader->m_Header.m_ModelVersion = 0;
	m_pHeader->m_Header.m_TrainingActive = 0;

	// Point to transition array
	m_pTransitions = reinterpret_cast<STransition *>(
		static_cast<char *>(m_pMemory) + sizeof(STrainingSharedMemory));

	log_info("rl/shm", "Created shared memory '%s' with capacity %zu transitions (%.2f MB)",
		pName, Capacity, m_MemorySize / (1024.0 * 1024.0));

	return true;
}

bool CTrainingSharedMemory::Open(const char *pName)
{
	if(m_pMemory)
	{
		log_error("rl/shm", "Shared memory already initialized");
		return false;
	}

	str_copy(m_Name, pName);
	m_IsOwner = false;

#ifdef CONF_FAMILY_UNIX
	// Open existing shared memory
	m_Fd = shm_open(pName, O_RDWR, 0666);
	if(m_Fd == -1)
	{
		log_error("rl/shm", "Failed to open shared memory '%s': %s", pName, strerror(errno));
		return false;
	}

	// Get size
	struct stat Stat;
	if(fstat(m_Fd, &Stat) == -1)
	{
		log_error("rl/shm", "Failed to get shared memory size: %s", strerror(errno));
		close(m_Fd);
		m_Fd = -1;
		return false;
	}
	m_MemorySize = Stat.st_size;

	// Map into address space
	m_pMemory = mmap(nullptr, m_MemorySize, PROT_READ | PROT_WRITE, MAP_SHARED, m_Fd, 0);
	if(m_pMemory == MAP_FAILED)
	{
		log_error("rl/shm", "Failed to map shared memory: %s", strerror(errno));
		close(m_Fd);
		m_pMemory = nullptr;
		m_Fd = -1;
		return false;
	}
#endif

#ifdef CONF_FAMILY_WINDOWS
	HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, pName);
	if(hMapFile == nullptr)
	{
		log_error("rl/shm", "Failed to open file mapping '%s': %lu", pName, GetLastError());
		return false;
	}

	m_Fd = (int)(intptr_t)hMapFile;

	// Map view - we need to read the header first to know the full size
	// Start with a minimal mapping to read the header
	m_pMemory = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	if(m_pMemory == nullptr)
	{
		log_error("rl/shm", "Failed to map view of file: %lu", GetLastError());
		CloseHandle(hMapFile);
		m_Fd = -1;
		return false;
	}

	// Get the actual size from the header
	m_pHeader = static_cast<STrainingSharedMemory *>(m_pMemory);
	m_MemorySize = CalculateMemorySize(m_pHeader->m_Header.m_Capacity);
#endif

	// Set up pointers
	m_pHeader = static_cast<STrainingSharedMemory *>(m_pMemory);
	m_pTransitions = reinterpret_cast<STransition *>(
		static_cast<char *>(m_pMemory) + sizeof(STrainingSharedMemory));

	log_info("rl/shm", "Opened shared memory '%s' with capacity %u transitions",
		pName, m_pHeader->m_Header.m_Capacity);

	return true;
}

void CTrainingSharedMemory::Close()
{
	if(!m_pMemory)
		return;

#ifdef CONF_FAMILY_UNIX
	munmap(m_pMemory, m_MemorySize);
	if(m_Fd != -1)
	{
		close(m_Fd);
	}
#endif

#ifdef CONF_FAMILY_WINDOWS
	UnmapViewOfFile(m_pMemory);
	if(m_Fd != -1)
	{
		CloseHandle((HANDLE)(intptr_t)m_Fd);
	}
#endif

	m_pMemory = nullptr;
	m_pHeader = nullptr;
	m_pTransitions = nullptr;
	m_Fd = -1;
	m_MemorySize = 0;
}

void CTrainingSharedMemory::Unlink()
{
	if(m_Name[0] == '\0')
		return;

#ifdef CONF_FAMILY_UNIX
	shm_unlink(m_Name);
#endif

	log_info("rl/shm", "Unlinked shared memory '%s'", m_Name);
}

STransition *CTrainingSharedMemory::GetTransition(size_t Index)
{
	if(!m_pTransitions)
		return nullptr;
	return &m_pTransitions[Index % m_pHeader->m_Header.m_Capacity];
}

const STransition *CTrainingSharedMemory::GetTransition(size_t Index) const
{
	if(!m_pTransitions)
		return nullptr;
	return &m_pTransitions[Index % m_pHeader->m_Header.m_Capacity];
}

bool CTrainingSharedMemory::WriteTransition(const STransition &Transition)
{
	if(!m_pHeader || !m_pTransitions)
		return false;

	// Get write index (atomic)
	uint64_t WriteIdx = __atomic_load_n(&m_pHeader->m_Header.m_WriteIndex, __ATOMIC_RELAXED);

	// Write the transition
	STransition *pDst = GetTransition(WriteIdx);
	*pDst = Transition;

	// Increment write index (atomic)
	__atomic_store_n(&m_pHeader->m_Header.m_WriteIndex, WriteIdx + 1, __ATOMIC_RELEASE);
	__atomic_fetch_add(&m_pHeader->m_Header.m_TotalWritten, 1, __ATOMIC_RELAXED);

	return true;
}

size_t CTrainingSharedMemory::WriteTransitions(const STransition *pTransitions, size_t Count)
{
	if(!m_pHeader || !m_pTransitions || Count == 0)
		return 0;

	uint64_t WriteIdx = __atomic_load_n(&m_pHeader->m_Header.m_WriteIndex, __ATOMIC_RELAXED);

	for(size_t i = 0; i < Count; i++)
	{
		STransition *pDst = GetTransition(WriteIdx + i);
		*pDst = pTransitions[i];
	}

	__atomic_store_n(&m_pHeader->m_Header.m_WriteIndex, WriteIdx + Count, __ATOMIC_RELEASE);
	__atomic_fetch_add(&m_pHeader->m_Header.m_TotalWritten, Count, __ATOMIC_RELAXED);

	return Count;
}

size_t CTrainingSharedMemory::ReadTransitions(STransition *pTransitions, size_t MaxCount)
{
	if(!m_pHeader || !m_pTransitions || MaxCount == 0)
		return 0;

	uint64_t WriteIdx = __atomic_load_n(&m_pHeader->m_Header.m_WriteIndex, __ATOMIC_ACQUIRE);
	uint64_t ReadIdx = __atomic_load_n(&m_pHeader->m_Header.m_ReadIndex, __ATOMIC_RELAXED);

	// Check for wraparound (buffer overflow)
	uint64_t Capacity = m_pHeader->m_Header.m_Capacity;
	if(WriteIdx - ReadIdx > Capacity)
	{
		// Data was overwritten, skip to latest available
		ReadIdx = WriteIdx - Capacity;
		log_warn("rl/shm", "Buffer overflow detected, skipping %lu transitions",
			(unsigned long)(WriteIdx - ReadIdx - Capacity));
	}

	size_t Available = static_cast<size_t>(WriteIdx - ReadIdx);
	size_t ToRead = (Available < MaxCount) ? Available : MaxCount;

	for(size_t i = 0; i < ToRead; i++)
	{
		pTransitions[i] = *GetTransition(ReadIdx + i);
	}

	__atomic_store_n(&m_pHeader->m_Header.m_ReadIndex, ReadIdx + ToRead, __ATOMIC_RELEASE);
	__atomic_fetch_add(&m_pHeader->m_Header.m_TotalRead, ToRead, __ATOMIC_RELAXED);

	return ToRead;
}

size_t CTrainingSharedMemory::GetAvailableTransitions() const
{
	if(!m_pHeader)
		return 0;

	uint64_t WriteIdx = __atomic_load_n(&m_pHeader->m_Header.m_WriteIndex, __ATOMIC_ACQUIRE);
	uint64_t ReadIdx = __atomic_load_n(&m_pHeader->m_Header.m_ReadIndex, __ATOMIC_RELAXED);

	return static_cast<size_t>(WriteIdx - ReadIdx);
}

uint64_t CTrainingSharedMemory::GetTotalWritten() const
{
	if(!m_pHeader)
		return 0;
	return __atomic_load_n(&m_pHeader->m_Header.m_TotalWritten, __ATOMIC_RELAXED);
}

uint64_t CTrainingSharedMemory::GetTotalRead() const
{
	if(!m_pHeader)
		return 0;
	return __atomic_load_n(&m_pHeader->m_Header.m_TotalRead, __ATOMIC_RELAXED);
}

SModelReloadSignal *CTrainingSharedMemory::GetModelSignal()
{
	if(!m_pHeader)
		return nullptr;
	return &m_pHeader->m_ModelSignal;
}

void CTrainingSharedMemory::SignalNewModel(const char *pModelPath)
{
	if(!m_pHeader)
		return;

	SModelReloadSignal *pSignal = &m_pHeader->m_ModelSignal;

	// Copy model path
	str_copy(pSignal->m_ModelPath, pModelPath);

	// Clear loaded flag and set ready flag
	__atomic_store_n(&pSignal->m_ModelLoaded, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&pSignal->m_NewModelReady, 1, __ATOMIC_RELEASE);
}

bool CTrainingSharedMemory::IsModelLoaded() const
{
	if(!m_pHeader)
		return false;
	return __atomic_load_n(&m_pHeader->m_ModelSignal.m_ModelLoaded, __ATOMIC_ACQUIRE) != 0;
}

void CTrainingSharedMemory::IncrementEpisodeCount()
{
	if(!m_pHeader)
		return;
	__atomic_fetch_add(&m_pHeader->m_Header.m_EpisodeCount, 1, __ATOMIC_RELAXED);
}

void CTrainingSharedMemory::AddSteps(uint64_t Steps)
{
	if(!m_pHeader)
		return;
	__atomic_fetch_add(&m_pHeader->m_Header.m_StepCount, Steps, __ATOMIC_RELAXED);
}

void CTrainingSharedMemory::AddReward(double Reward)
{
	if(!m_pHeader)
		return;
	// Note: This is not atomic for double, but good enough for statistics
	m_pHeader->m_Header.m_TotalReward += Reward;
}

void CTrainingSharedMemory::SetTrainingActive(bool Active)
{
	if(!m_pHeader)
		return;
	__atomic_store_n(&m_pHeader->m_Header.m_TrainingActive, Active ? 1 : 0, __ATOMIC_RELEASE);
}

uint64_t CTrainingSharedMemory::GetEpisodeCount() const
{
	if(!m_pHeader)
		return 0;
	return __atomic_load_n(&m_pHeader->m_Header.m_EpisodeCount, __ATOMIC_RELAXED);
}

uint64_t CTrainingSharedMemory::GetStepCount() const
{
	if(!m_pHeader)
		return 0;
	return __atomic_load_n(&m_pHeader->m_Header.m_StepCount, __ATOMIC_RELAXED);
}

double CTrainingSharedMemory::GetTotalReward() const
{
	if(!m_pHeader)
		return 0.0;
	return m_pHeader->m_Header.m_TotalReward;
}

bool CTrainingSharedMemory::IsTrainingActive() const
{
	if(!m_pHeader)
		return false;
	return __atomic_load_n(&m_pHeader->m_Header.m_TrainingActive, __ATOMIC_ACQUIRE) != 0;
}

size_t CTrainingSharedMemory::GetCapacity() const
{
	if(!m_pHeader)
		return 0;
	return m_pHeader->m_Header.m_Capacity;
}

} // namespace rl
