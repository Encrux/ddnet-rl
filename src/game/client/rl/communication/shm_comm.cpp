/* Teeworlds RL Bot - POSIX Shared Memory Communication Implementation */
#include "shm_comm.h"

#include <base/system.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>

namespace rl {

CShmComm::CShmComm() :
	m_ShmFd(-1),
	m_pShmPtr(nullptr),
	m_ShmSize(0),
	m_FrameBufferSize(0),
	m_FrameWidth(0),
	m_FrameHeight(0),
	m_FrameChannels(0),
	m_Initialized(false)
{
	m_aShmPath[0] = '\0';
}

CShmComm::~CShmComm()
{
	Shutdown();
}

bool CShmComm::Init(int EnvId, uint32_t FrameWidth, uint32_t FrameHeight, uint32_t FrameChannels)
{
	if(m_Initialized)
	{
		dbg_msg("rl/shm", "Already initialized");
		return false;
	}
	
	m_FrameWidth = FrameWidth;
	m_FrameHeight = FrameHeight;
	m_FrameChannels = FrameChannels;
	m_FrameBufferSize = (size_t)FrameWidth * FrameHeight * FrameChannels;
	
	// Total size: header + frame buffer
	m_ShmSize = sizeof(SSharedMemoryHeader) + m_FrameBufferSize;
	
	// Create shared memory name
	str_format(m_aShmPath, sizeof(m_aShmPath), "/teeworlds_rl_%d", EnvId);
	
	// Remove existing shared memory if it exists
	shm_unlink(m_aShmPath);
	
	// Create shared memory object
	m_ShmFd = shm_open(m_aShmPath, O_CREAT | O_RDWR, 0666);
	if(m_ShmFd == -1)
	{
		dbg_msg("rl/shm", "Failed to create shared memory: %s", m_aShmPath);
		return false;
	}
	
	// Set size
	if(ftruncate(m_ShmFd, m_ShmSize) == -1)
	{
		dbg_msg("rl/shm", "Failed to set shared memory size");
		close(m_ShmFd);
		shm_unlink(m_aShmPath);
		return false;
	}
	
	// Map shared memory
	m_pShmPtr = mmap(nullptr, m_ShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, m_ShmFd, 0);
	if(m_pShmPtr == MAP_FAILED)
	{
		dbg_msg("rl/shm", "Failed to map shared memory");
		close(m_ShmFd);
		shm_unlink(m_aShmPath);
		m_pShmPtr = nullptr;
		return false;
	}
	
	// Initialize header
	SSharedMemoryHeader *pHeader = GetHeader();
	mem_zero(pHeader, sizeof(SSharedMemoryHeader));
	pHeader->m_FrameWidth = FrameWidth;
	pHeader->m_FrameHeight = FrameHeight;
	pHeader->m_FrameChannels = FrameChannels;
	
	m_Initialized = true;
	dbg_msg("rl/shm", "Shared memory initialized: %s (size: %zu bytes, frame: %ux%ux%u)", 
		m_aShmPath, m_ShmSize, FrameWidth, FrameHeight, FrameChannels);
	
	return true;
}

void CShmComm::Shutdown()
{
	if(!m_Initialized)
		return;
	
	// Signal shutdown to Python
	if(m_pShmPtr)
	{
		SSharedMemoryHeader *pHeader = GetHeader();
		pHeader->m_Shutdown = 1;
		pHeader->m_ClientReady = 1; // Wake up Python if waiting
	}
	
	// Unmap and close
	if(m_pShmPtr && m_pShmPtr != MAP_FAILED)
	{
		munmap(m_pShmPtr, m_ShmSize);
		m_pShmPtr = nullptr;
	}
	
	if(m_ShmFd != -1)
	{
		close(m_ShmFd);
		m_ShmFd = -1;
	}
	
	// Remove shared memory
	if(m_aShmPath[0] != '\0')
	{
		shm_unlink(m_aShmPath);
	}
	
	m_Initialized = false;
	dbg_msg("rl/shm", "Shared memory shutdown complete");
}

SSharedMemoryHeader *CShmComm::GetHeader()
{
	return static_cast<SSharedMemoryHeader *>(m_pShmPtr);
}

uint8_t *CShmComm::GetFrameBuffer()
{
	if(!m_pShmPtr)
		return nullptr;
	return static_cast<uint8_t *>(m_pShmPtr) + sizeof(SSharedMemoryHeader);
}

size_t CShmComm::GetFrameBufferSize() const
{
	return m_FrameBufferSize;
}

void CShmComm::SignalFrameReady()
{
	if(!m_pShmPtr)
		return;
	
	SSharedMemoryHeader *pHeader = GetHeader();
	
	// Memory barrier to ensure frame data is written before flag
	__sync_synchronize();
	
	pHeader->m_ClientReady = 1;
}

bool CShmComm::IsActionReady()
{
	if(!m_pShmPtr)
		return false;
	
	SSharedMemoryHeader *pHeader = GetHeader();
	return pHeader->m_PythonReady != 0;
}

void CShmComm::ClearActionReady()
{
	if(!m_pShmPtr)
		return;
	
	SSharedMemoryHeader *pHeader = GetHeader();
	pHeader->m_PythonReady = 0;
	pHeader->m_ClientReady = 0;
}

bool CShmComm::IsShutdownRequested()
{
	if(!m_pShmPtr)
		return false;
	
	SSharedMemoryHeader *pHeader = GetHeader();
	return pHeader->m_Shutdown != 0;
}

} // namespace rl
