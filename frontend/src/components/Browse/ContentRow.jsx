import { useState, useRef } from 'react'
import { FaChevronLeft, FaChevronRight } from 'react-icons/fa'
import ContentCard from './ContentCard'

export default function ContentRow({ title, items }) {
  const [scrollPosition, setScrollPosition] = useState(0)
  const rowRef = useRef(null)

  const scroll = (direction) => {
    const container = rowRef.current
    if (!container) return

    const scrollAmount = container.offsetWidth * 0.8
    const newPosition = direction === 'left' 
      ? Math.max(0, scrollPosition - scrollAmount)
      : scrollPosition + scrollAmount

    container.scrollTo({
      left: newPosition,
      behavior: 'smooth',
    })
    setScrollPosition(newPosition)
  }

  return (
    <div className="px-12 group">
      <h2 className="text-2xl font-bold mb-4 hover:text-gray-300 transition-colors cursor-pointer">
        {title}
      </h2>

      <div className="relative">
        {/* Left Arrow */}
        <button
          onClick={() => scroll('left')}
          className="absolute left-0 top-0 bottom-0 z-20 w-12 bg-black/50 hover:bg-black/80 text-white opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
        >
          <FaChevronLeft size={24} />
        </button>

        {/* Content Carousel */}
        <div
          ref={rowRef}
          className="flex gap-2 overflow-x-scroll scrollbar-hide scroll-smooth"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {items.map((item, index) => (
            <ContentCard key={item.title + index} content={item} />
          ))}
        </div>

        {/* Right Arrow */}
        <button
          onClick={() => scroll('right')}
          className="absolute right-0 top-0 bottom-0 z-20 w-12 bg-black/50 hover:bg-black/80 text-white opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
        >
          <FaChevronRight size={24} />
        </button>
      </div>
    </div>
  )
}
