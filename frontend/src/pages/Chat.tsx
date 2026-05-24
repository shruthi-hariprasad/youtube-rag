import { useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import Navbar from "../components/Navbar"

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

interface Source {
  video_id: string
  title: string
  chunk_index: number
  text: string
}

interface Message {
  role: "user" | "assistant"
  content: string
  sources?: Source[]
}

interface Video {
  youtube_video_id: string
  title: string
  channel_name: string
  thumbnail_url: string
  url: string
}

export default function Chat() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const video: Video | null = state?.video ?? null

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())

  function toggleSources(idx: number) {
    setExpandedSources(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input
    setMessages(prev => [...prev, { role: "user", content: question }])
    setInput("")
    setLoading(true)
    setError("")

    // Optimistically add an empty assistant message that will be filled by the stream
    setMessages(prev => [...prev, { role: "assistant", content: "" }])

    try {
      const url = new URL(`${BASE_URL}/query/stream`)
      url.searchParams.set("question", question)
      if (video) url.searchParams.set("video_id", video.youtube_video_id)

      const token = localStorage.getItem("token")
      const response = await fetch(url.toString(), {
        headers: { Authorization: `Bearer ${token}` },
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Request failed")
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const data = JSON.parse(line.slice(6))
          if (data.token) {
            setMessages(prev => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              updated[updated.length - 1] = { ...last, content: last.content + data.token }
              return updated
            })
          } else if (data.done) {
            setMessages(prev => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              updated[updated.length - 1] = { ...last, sources: data.sources }
              return updated
            })
          }
        }
      }
    } catch (err: any) {
      setMessages(prev => prev.slice(0, -1))
      setError(err.message || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">

      <Navbar />

      {/* Sub-header: back button + video info */}
      <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-4">
        <button
          onClick={() => navigate("/library")}
          className="text-gray-400 hover:text-gray-600 text-sm"
        >
          ← Library
        </button>

        {video ? (
          <>
            <img
              src={video.thumbnail_url}
              alt={video.title}
              className="w-14 rounded-lg object-cover"
            />
            <div>
              <p className="text-sm font-semibold text-gray-900 line-clamp-1">{video.title}</p>
              <p className="text-xs text-gray-500">{video.channel_name}</p>
            </div>
          </>
        ) : (
          <div>
            <p className="text-sm font-semibold text-gray-900">Search your entire library</p>
            <p className="text-xs text-gray-500">Answers will include which video they came from</p>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 max-w-3xl w-full mx-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg mb-1">
              {video ? "Ask anything about this video" : "Ask anything across your library"}
            </p>
            <p className="text-sm">
              {video
                ? "The answer will be grounded in the transcript"
                : "The answer will tell you which video it came from"}
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i}>
            {msg.role === "user" ? (
              <div className="flex justify-end">
                <div className="bg-indigo-600 text-white text-sm px-4 py-3 rounded-2xl rounded-tr-sm max-w-xl">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <div className="bg-white border border-gray-200 text-gray-800 text-sm px-4 py-3 rounded-2xl rounded-tl-sm max-w-xl shadow-sm whitespace-pre-wrap">
                  {msg.content}
                  {loading && i === messages.length - 1 && msg.content === "" && (
                    <span className="inline-flex gap-1 pl-1">
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                    </span>
                  )}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="max-w-xl">
                    <button
                      onClick={() => toggleSources(i)}
                      className="text-xs text-gray-400 hover:text-gray-600 transition"
                    >
                      {expandedSources.has(i)
                        ? "Hide sources"
                        : `Show sources (${msg.sources.length})`}
                    </button>
                    {expandedSources.has(i) && (
                      <div className="space-y-2 mt-2">
                        {msg.sources.map((src, j) => (
                          <div key={j} className="bg-gray-100 rounded-lg px-4 py-2.5 text-xs text-gray-600">
                            <p className="font-medium text-gray-700 mb-1">{src.title} · chunk {src.chunk_index}</p>
                            <p className="line-clamp-3 leading-relaxed">{src.text}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {error && <p className="text-red-500 text-sm">{error}</p>}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <form onSubmit={handleAsk} className="max-w-3xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={video ? "Ask a question about this video..." : "Ask anything across your library..."}
            className="flex-1 border border-gray-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition disabled:opacity-50"
          >
            Ask
          </button>
        </form>
      </div>

    </div>
  )
}
