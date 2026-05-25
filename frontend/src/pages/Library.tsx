import { useState, useEffect, useRef } from "react"
import { useNavigate } from "react-router-dom"
import api from "../api/axios"
import Navbar from "../components/Navbar"

interface Video {
  id: number
  youtube_video_id: string
  title: string
  channel_name: string
  thumbnail_url: string
  url: string
  summary?: string | null
  suggested_questions?: string | null
}

const PROGRESS_STEPS = [
  "Fetching transcript...",
  "Processing chunks...",
  "Generating embeddings...",
  "Generating summary...",
]

export default function Library() {
  const [videos, setVideos] = useState<Video[]>([])
  const [url, setUrl] = useState("")
  const [loading, setLoading] = useState(false)
  const [fetching, setFetching] = useState(true)
  const [error, setError] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [progressStep, setProgressStep] = useState(0)
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const navigate = useNavigate()

  const filteredVideos = videos.filter(v => {
    const q = searchQuery.toLowerCase()
    return (
      v.title?.toLowerCase().includes(q) ||
      v.channel_name?.toLowerCase().includes(q)
    )
  })

  useEffect(() => {
    api.get("/videos")
      .then(res => setVideos(res.data))
      .catch(() => setError("Failed to load videos"))
      .finally(() => setFetching(false))
  }, [])

  async function handleDelete(e: React.MouseEvent, videoId: number) {
    e.stopPropagation()
    if (!window.confirm("Remove this video from your library?")) return
    try {
      await api.delete(`/videos/${videoId}`)
      setVideos(prev => prev.filter(v => v.id !== videoId))
    } catch {
      setError("Failed to delete video")
    }
  }

  async function handleAddVideo(e: React.FormEvent) {
    e.preventDefault()
    setError("")
    setLoading(true)
    setProgressStep(0)
    progressRef.current = setInterval(() => {
      setProgressStep(prev => (prev < PROGRESS_STEPS.length - 1 ? prev + 1 : prev))
    }, 4000)
    try {
      const res = await api.post(`/videos?url=${encodeURIComponent(url)}`)
      setVideos(prev => [res.data, ...prev])
      setUrl("")
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to add video")
    } finally {
      if (progressRef.current) clearInterval(progressRef.current)
      setProgressStep(0)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />

      {/* Header */}
      <div className="bg-white border-b border-gray-100">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <h1 className="text-2xl font-bold text-gray-900">Your library</h1>
          <p className="text-sm text-gray-500 mt-1">Add YouTube videos and ask questions about them</p>

          {/* Add video form */}
          <form onSubmit={handleAddVideo} className="flex gap-3 mt-5">
            <input
              type="text"
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder="Paste a YouTube URL..."
              className="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-gray-50 transition"
              required
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-xl text-sm transition disabled:opacity-60 whitespace-nowrap min-w-[170px]"
            >
              {loading ? PROGRESS_STEPS[progressStep] : "Add video"}
            </button>
          </form>

          {error && <p className="text-red-500 text-sm mt-3">{error}</p>}
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-6">
        {/* Toolbar */}
        {videos.length > 0 && (
          <div className="flex items-center gap-3 mb-6">
            <button
              onClick={() => navigate("/chat", { state: { video: null } })}
              className="flex items-center gap-2 bg-white border border-indigo-200 hover:border-indigo-400 text-indigo-600 font-medium px-4 py-2 rounded-xl text-sm transition shadow-sm"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-4 h-4">
                <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
              </svg>
              Search entire library
            </button>

            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Filter by title or channel..."
              className="flex-1 border border-gray-200 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white transition"
            />

            <span className="text-sm text-gray-400 whitespace-nowrap">
              {filteredVideos.length} video{filteredVideos.length !== 1 ? "s" : ""}
            </span>
          </div>
        )}

        {/* Content */}
        {fetching ? (
          <div className="flex items-center justify-center py-24 text-gray-400">
            <svg className="animate-spin w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Loading your library...
          </div>
        ) : videos.length === 0 ? (
          <div className="text-center py-24">
            <div className="w-16 h-16 bg-indigo-50 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth={1.5} className="w-8 h-8">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
            </div>
            <p className="text-gray-700 font-medium mb-1">No videos yet</p>
            <p className="text-sm text-gray-400">Paste a YouTube URL above to get started</p>
          </div>
        ) : filteredVideos.length === 0 ? (
          <p className="text-gray-400 text-sm py-8">No videos match "{searchQuery}"</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredVideos.map(video => (
              <div
                key={video.id}
                onClick={() => navigate("/chat", { state: { video } })}
                className="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-md ring-1 ring-gray-100 hover:ring-indigo-200 transition-all cursor-pointer group"
              >
                <div className="relative overflow-hidden">
                  <img
                    src={video.thumbnail_url}
                    alt={video.title}
                    className="w-full aspect-video object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                    <div className="w-10 h-10 bg-white/90 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-md">
                      <svg viewBox="0 0 24 24" fill="#6366f1" className="w-5 h-5 ml-0.5">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </div>
                  </div>
                </div>
                <div className="p-3.5 flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-gray-900 line-clamp-2 leading-snug">{video.title}</p>
                    <p className="text-xs text-gray-400 mt-1">{video.channel_name}</p>
                    {video.summary && (
                      <p className="text-xs text-gray-400 mt-2 line-clamp-2 leading-relaxed">{video.summary}</p>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, video.id)}
                    className="shrink-0 w-6 h-6 flex items-center justify-center text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition"
                    title="Remove video"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-3.5 h-3.5">
                      <path strokeLinecap="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
